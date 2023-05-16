import argparse, os, sys, glob, re
from collections import deque
import cv2
import torch
import torch.nn as nn
import safetensors.torch
import numpy as np
import random
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
#from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast as ac
from contextlib import contextmanager, nullcontext
import accelerate
import k_diffusion as K
from scripts import prompt_parser
import inspect

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
NEAREST = (Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST)

#from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
#safety_model_id = "CompVis/stable-diffusion-safety-checker"
#safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
#safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
prompt_filter_regex = r'[\(\)]|:\d+(\.\d+)?'

k_diffusion = [
    'euler_a', 'euler','lms', 'heun', 'dpm_2', 'dpm_2_a', 'dpmpp_2s_a', 'dpmpp_2m', 'dpmpp_sde',
    'dpm_fast', 'dpm_ad', 'lms_ka', 'dpm_2_ka', 'dpm_2_a_ka', 'dpmpp_2s_a_ka', 'dpmpp_2m_ka', 'dpmpp_sde_ka', 
]
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def txt2img_image_conditioning(sd_model, x, width, height, device):
    if sd_model.model.conditioning_key in {'hybrid', 'concat'}: # Inpainting models

        # The "masked-image" in this case will just be all zeros since the entire image is masked.
        image_conditioning = torch.zeros(x.shape[0], 3, height, width, device=device)
        image_conditioning = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(image_conditioning))

        # Add the fake full 1s mask to the first dimension.
        image_conditioning = torch.nn.functional.pad(image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0)
        image_conditioning = image_conditioning.to(x.dtype)

        return image_conditioning

    elif sd_model.model.conditioning_key == "crossattn-adm": # UnCLIP models

        return x.new_zeros(x.shape[0], 2*sd_model.noise_augmentor.time_embed.dim, dtype=x.dtype, device=x.device)

    else:
        # Dummy zero conditioning if we're not using inpainting or unclip models.
        # Still takes up a bit of memory, but no encoder call.
        # Pretty sure we can just make this a 1x1 image since its not going to be used besides its batch size.
        return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)
def upscale(img, scale):
    dest_w = int(img.width * scale)
    dest_h = int(img.height * scale)

    for i in range(3):
        shape = (img.width, img.height)
        
        if shape == (img.width, img.height):
            break

        if img.width >= dest_w and img.height >= dest_h:
            break

    if img.width != dest_w or img.height != dest_h:
        img = img.resize((int(dest_w), int(dest_h)), resample=NEAREST)

    return img    
class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.step = 0
    def forward(self, x, sigma, uncond, cond, cond_scale):
        #image_uncond = image_cond
        batch_size = len(cond)
        repeats = [len(cond[i]) for i in range(batch_size)]
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        #image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_uncond])
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale
    #def forward(self, x, sigma, uncond, cond, cond_scale):
    #    conds_list, tensor = prompt_parser.reconstruct_multicond_batch(cond, self.step)
    #    uncond = prompt_parser.reconstruct_cond_batch(uncond, self.step)
    #    batch_size = len(conds_list)
    #    repeats = [len(conds_list[i]) for i in range(batch_size)]
    #    x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x])
    #    sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma])
    #    cond_in = torch.cat([uncond, tensor])
    #    uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
    #    self.step += 1
    #    return uncond + (cond - uncond) * cond_scale

    
checkpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

checkpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]

def transform_checkpoint_dict_key(k):
  for text, replacement in checkpoint_dict_replacements.items():
      if k.startswith(text):
          k = replacement + k[len(text):]

  return k

def get_state_dict_from_checkpoint(pl_sd):
  pl_sd = pl_sd.pop("state_dict", pl_sd)
  pl_sd.pop("state_dict", None)

  sd = {}
  for k, v in pl_sd.items():
      new_key = transform_checkpoint_dict_key(k)

      if new_key is not None:
          sd[new_key] = v

  pl_sd.clear()
  pl_sd.update(sd)

  return pl_sd

def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
  _, extension = os.path.splitext(checkpoint_file)
  if extension.lower() == ".safetensors":
      device = map_location
      pl_sd = safetensors.torch.load_file(checkpoint_file, device=device)
  else:
      pl_sd = torch.load(checkpoint_file, map_location=map_location)

  if print_global_state and "global_step" in pl_sd:
      print(f"Global Step: {pl_sd['global_step']}")

  sd = get_state_dict_from_checkpoint(pl_sd)
  return sd

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    _, extension = os.path.splitext(ckpt)
    if extension.lower() == ".safetensors":
        device = "cpu"
        pl_sd = safetensors.torch.load_file(ckpt, device=device)
    else:
        pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = read_state_dict(ckpt, map_location="cpu")
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    
    model.cuda()
    sd_hijack.hijack(model)
    model.eval()
    return model


#def put_watermark(img, wm_encoder=None):
#    if wm_encoder is not None:
#        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#        img = wm_encoder.encode(img, 'dwtDct')
#        img = Image.fromarray(img[:, :, ::-1])
#    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def resize_image(source_image, destination_image, width, height, resize_factor):
    image_to_resize = cv2.imread(source_image)
    resized_image = cv2.resize(image_to_resize, dsize=(width*resize_factor, height*resize_factor)
                               , interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(destination_image, resized_image)

def upscale_image(source_image, destination_image, width, height, resize_factor):
    image_to_resize = cv2.imread(source_image)
    resized_image = cv2.resize(image_to_resize, dsize=(width*resize_factor, height*resize_factor)
                               , interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(destination_image, resized_image)

def improve_image(source_image, destination_image):
    image_to_improve = cv2.imread(source_image)
    image_improved = cv2.detailEnhance(image_to_improve, sigma_s=200, sigma_r=0.01)
    image_improved = cv2.edgePreservingFilter(image_improved, flags=1, sigma_s=60, sigma_r=0.3)
    cv2.imwrite(destination_image, image_improved)

def merge_prompts(style_prompt: str, prompt: str) -> str:
    if "{prompt}" in style_prompt:
        res = style_prompt.replace("{prompt}", prompt)
    else:
        parts = filter(None, (prompt.strip(), style_prompt.strip()))
        res = ", ".join(parts)

    return res

def apply_styles_to_prompt(prompt, styles):
    for style in styles:
        prompt = merge_prompts(style, prompt)

    return prompt

def read_prompt_parameter(parser):
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

def read_negative_prompt_parameter(parser):
    parser.add_argument(
        "--negative_prompt",
        type=str,
        nargs="?",
        default="worst quality",
        help="the negative prompt to render"
    )

def read_outdir_parameter(parser):
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )


def read_skip_grid_parameter(parser):
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )


def read_skip_save_parameter(parser):
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )


def read_ddim_steps_parameter(parser):
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

def read_sampler(parser):
    parser.add_argument(
        "--sampler",
        type=str,
        default="DDIM",
        help="select sampler",
    )

def read_plms_parameter(parser):
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )


def read_dpm_solver_parameter(parser):
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )


def read_laion400m_parameter(parser):
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )


def read_fixed_code_parameter(parser):
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )


def read_ddim_eta_parameter(parser):
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )


def read_n_iter_parameter(parser):
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )


def read_height_parameter(parser):
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )


def read_width_parameter(parser):
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )


def read_latent_channels_parameter(parser):
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )


def read_downsampling_factor_parameter(parser):
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )


def read_n_samples_parameter(parser):
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )


def read_n_rows_parameter(parser):
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )


def read_scale_parameter(parser):
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )


def read_from_file_parameter(parser):
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )


def read_config_parameter(parser):
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )


def read_ckpt_parameter(parser):
    parser.add_argument(
        "--model",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )


def read_seed_parameter(parser):
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(1, 4294967295),
        help="the seed (for reproducible sampling)",
    )


def read_precision_parameter(parser):
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

def read_quantize(parser):
    parser.add_argument(
        "--quantize",
        action='store_true',
        help="quantize",
    )

def read_resize_factor_parameter(parser):
    parser.add_argument(
        "--resize_factor",
        type=float,
        help="Resize factor",
        default=2.0
    )
def read_hr_steps(parser):
    parser.add_argument(
        "--hr_steps",
        type=int,
        default=0,
        help="number of high resolutionize sampling steps",
    )
def read_denoise_strength(parser):
    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
def setup_img2img_steps(p, steps=None):
    if steps is not None:
        requested_steps = (steps or p.ddim_steps)
        steps = int(requested_steps / min(p.strength, 0.999)) if p.strength > 0 else 0
        t_enc = requested_steps - 1
    else:
        steps = p.ddim_steps
        t_enc = int(min(p.strength, 0.999) * steps)

    return steps, t_enc
class TorchHijack:
    def __init__(self, sampler_noises):
        # Using a deque to efficiently receive the sampler_noises in the same order as the previous index-based
        # implementation.
        self.sampler_noises = deque(sampler_noises)

    def __getattr__(self, item):
        if item == 'randn_like':
            return self.randn_like

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

    def randn_like(self, x):
        if self.sampler_noises:
            noise = self.sampler_noises.popleft()
            if noise.shape == x.shape:
                return noise
        return torch.randn_like(x)

def main():
    cached_uc = [None, None]
    cached_c = [None, None]
    
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    read_prompt_parameter(parser)
    read_negative_prompt_parameter(parser)
    read_outdir_parameter(parser)
    read_skip_grid_parameter(parser)
    read_skip_save_parameter(parser)
    read_ddim_steps_parameter(parser)
    read_hr_steps(parser)
    read_denoise_strength(parser)
    read_sampler(parser)
    read_laion400m_parameter(parser)
    read_fixed_code_parameter(parser)
    read_ddim_eta_parameter(parser)
    read_n_iter_parameter(parser)
    read_height_parameter(parser)
    read_width_parameter(parser)
    read_latent_channels_parameter(parser)
    read_quantize(parser)
    read_downsampling_factor_parameter(parser)
    read_n_samples_parameter(parser)
    read_n_rows_parameter(parser)
    read_scale_parameter(parser)
    read_from_file_parameter(parser)
    read_config_parameter(parser)
    read_ckpt_parameter(parser)
    read_seed_parameter(parser)
    read_precision_parameter(parser)
    read_resize_factor_parameter(parser)
    
    opt = parser.parse_args()
    
    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.model = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"
    if opt.seed == -1:
      seed_f = int(random.randrange(4294967294))
    else:
      seed_f = opt.seed
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.model}")
    model = model.half()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    accelerator = accelerate.Accelerator()
    model = model.to(device)
    k_d = False
    karras = False
    
    if opt.sampler=="DPMS":
        sampler = DPMSolverSampler(model)
    elif opt.sampler=="PLMS":
        sampler = PLMSSampler(model)
    elif opt.sampler=="DDIM":
        sampler = DDIMSampler(model)
    elif opt.sampler in k_diffusion:
        model_wrap = K.external.CompVisVDenoiser(model, quantize=opt.quantize) if model.parameterization == "v" else K.external.CompVisDenoiser(model, quantize=opt.quantize)
        sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()
        k_d = True
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    #print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    #wm = "StableDiffusionV1"
    #wm_encoder = WatermarkEncoder()
    #wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
    steps = opt.ddim_steps
    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    sampler_noises = None
    
    def slerp(val, low, high):
        low_norm = low/torch.norm(low, dim=1, keepdim=True)
        high_norm = high/torch.norm(high, dim=1, keepdim=True)
        dot = (low_norm*high_norm).sum(1)

        if dot.mean() > 0.9995:
            return low * val + high * (1 - val)

        omega = torch.acos(dot)
        so = torch.sin(omega)
        res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
        return res
    def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, device=device):
        global sampler_noises
        eta_noise_seed_delta = opt.ddim_eta or 0
        xs = []

        # if we have multiple seeds, this means we are working with batch size>1; this then
        # enables the generation of additional tensors with noise that the sampler will use during its processing.
        # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
        # produce the same images as with two batches [100], [101].
        ase = steps if k_d else 0
        if len(seeds) > 1:
            sampler_noises = [[] for _ in range(ase)]
        else:
            sampler_noises = None

        for i, seed in enumerate(seeds):
            noise_shape = shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (shape[0], seed_resize_from_h//8, seed_resize_from_w//8)

            subnoise = None
            if subseeds is not None:
                subseed = 0 if i >= len(subseeds) else subseeds[i]
                torch.manual_seed(subseed)
                subnoise = torch.randn(noise_shape, device=device)

            # randn results depend on device; gpu and cpu get different results for same seed;
            # the way I see it, it's better to do this on CPU, so that everyone gets same result;
            # but the original script had it like this, so I do not dare change it for now because
            # it will break everyone's seeds.
            torch.manual_seed(seed)
            noise = torch.randn(noise_shape, device=device)

            if subnoise is not None:
                noise = slerp(subseed_strength, noise, subnoise)

            if noise_shape != shape:
                torch.manual_seed(seed)
                x = torch.randn(shape, device=device)
                dx = (shape[2] - noise_shape[2]) // 2
                dy = (shape[1] - noise_shape[1]) // 2
                w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
                h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
                tx = 0 if dx < 0 else dx
                ty = 0 if dy < 0 else dy
                dx = max(-dx, 0)
                dy = max(-dy, 0)

                x[:, ty:ty+h, tx:tx+w] = noise[:, dy:dy+h, dx:dx+w]
                noise = x

            if sampler_noises is not None:
                cnt = ase

                if eta_noise_seed_delta > 0:
                    torch.manual_seed(seed + eta_noise_seed_delta)

                for j in range(cnt):
                    sampler_noises[j].append(torch.randn(tuple(noise_shape),device=device))

            xs.append(noise)

        if sampler_noises is not None:
            amber = [torch.stack(n).to(device) for n in sampler_noises]
            sampler_noises = amber

        x = torch.stack(xs).to(device)
        return x
                                    
    if not opt.from_file:
        prompt = opt.prompt
        negative_prompt = opt.negative_prompt
        if type(prompt) == list:
            all_prompts = prompt
        else:
            all_prompts = batch_size * opt.n_iter * [prompt]
        if type(negative_prompt) == list:
            all_negative_prompts = negative_prompt
        else:
            all_negative_prompts = batch_size * opt.n_iter * [negative_prompt]
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    # Folder with the original output
    os.makedirs(os.path.join(sample_path, "original"), exist_ok=True)
    # Folder with the resized output
    os.makedirs(os.path.join(sample_path, "resized"), exist_ok=True)
    # Folder with the improved output based on the resized output
    os.makedirs(os.path.join(sample_path, "improved"), exist_ok=True)
    base_count = len(os.listdir(os.path.join(sample_path, "original")))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = ac if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                def autocast(disable=False):
                    if disable:
                        return contextlib.nullcontext()

                    if opt.precision == "full":
                        return nullcontext()

                    return ac("cuda")
                def decode_first_stage(model, x):
                    with autocast():
                        x = model.decode_first_stage(x)

                    return x
                
                def get_conds_with_caching(function, required_prompts, steps, cache):
                    if cache[0] is not None and (required_prompts, steps) == cache[0]:
                        return cache[1]

                    with autocast():
                        cache[1] = function(model, required_prompts, steps)

                    cache[0] = (required_prompts, steps)
                    return cache[1]
                for n in trange(opt.n_iter, desc="Sampling", disable =not accelerator.is_main_process):
                    prompts = all_prompts[n * batch_size:(n + 1) * batch_size]
                    negative_prompts = all_negative_prompts[n * batch_size:(n + 1) * batch_size]
                    uc = None
                    step_multiplier = 2 if opt.sampler in ['dpmpp_2s_a', 'dpmpp_2s_a_ka', 'dpmpp_sde', 'dpmpp_sde_ka', 'dpm_2', 'dpm_2_a', 'heun'] else 1
                    if opt.scale != 1.0:
                        uc = torch.cat([prompt_parser.get_learned_conditioning_with_prompt_weights(nega_pr, model) for nega_pr in negative_prompts])
                        #uc = get_conds_with_caching(prompt_parser.get_learned_conditioning, negative_prompts, steps * step_multiplier, cached_uc)
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    #c = get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, steps * step_multiplier, cached_c)
                    c = torch.cat([prompt_parser.get_learned_conditioning_with_prompt_weights(pr, model) for pr in prompts])
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    if k_d:
                        def create_noise_sampler(x, sigmas, opt):
                            from k_diffusion.sampling import BrownianTreeNoiseSampler
                            sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
                            if type(opt.seed) == list:
                                all_seeds = opt.seed
                            else:
                                all_seeds = [int(opt.seed) + a for a in range(len(all_prompts))]
                            current_iter_seeds = all_seeds[n * opt.n_samples:(n + 1) * opt.n_samples]
                            return BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=current_iter_seeds)
                        if opt.sampler.endswith("_ka"):
                            sigma_min, sigma_max = (model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item())
                            sigmas = K.sampling.get_sigmas_karras(opt.ddim_steps, sigma_min=sigma_min, sigma_max=sigma_max, device=device)
                            opt.sampler = opt.sampler[:-3]
                            karras = True
                        else:
                            sigmas = model_wrap.get_sigmas(opt.ddim_steps)
                        if "dpm_2" in opt.sampler:
                            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
                        torch.manual_seed(opt.seed) # changes manual seeding procedure
                        #x = torch.randn([opt.n_samples, *shape], device=device) * sigmas[0] # for GPU draw
                        
                        x = create_random_tensors(shape, seeds=[seed_f])
                        #image_cond = x.new_zeros(x.shape[0], 5, 1, 1, dtype=torch.float16, device=device)
                        x *= sigmas[0]
                        def initialize():

                            K.sampling.torch = TorchHijack(sampler_noises if sampler_noises is not None else [])

                            extra_params_kwargs = {}

                            if 'eta' in inspect.signature(K.sampling.__dict__[f'sample_{opt.sampler}']).parameters:
                                extra_params_kwargs['eta'] = 1.0

                            return extra_params_kwargs
                        K.sampling.torch = TorchHijack(sampler_noises if sampler_noises is not None else [])
                        extra_params_kwargs = initialize()
                        parameters = inspect.signature(K.sampling.__dict__[f'sample_{opt.sampler}']).parameters
                        if 'sigma_min' in parameters:
                            extra_params_kwargs['sigma_min'] = model_wrap.sigmas[0].item()
                            extra_params_kwargs['sigma_max'] = model_wrap.sigmas[-1].item()
                            if 'n' in parameters:
                                extra_params_kwargs['n'] = steps
                        else:
                            extra_params_kwargs['sigmas'] = sigmas
                        if "dpmpp_sde" in opt.sampler:
                            noise_sampler = create_noise_sampler(x, sigmas, opt)
                            extra_params_kwargs['noise_sampler'] = noise_sampler
                        # x = torch.randn([opt.n_samples, *shape]).to(device) * sigmas[0] # for CPU draw
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {'cond': c,  'uncond': uc, 'cond_scale': opt.scale}
                        
                        samples_ddim = K.sampling.__dict__[f'sample_{opt.sampler}'](model_wrap_cfg, x, extra_args=extra_args, disable=not accelerator.is_main_process, **extra_params_kwargs)
                        if karras:
                            opt.sampler = opt.sampler + "_ka"
                            karras = False
                    else:
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                          conditioning=c,
                                                          batch_size=opt.n_samples,
                                                          shape=shape,
                                                          verbose=False,
                                                          unconditional_guidance_scale=opt.scale,
                                                          unconditional_conditioning=uc,
                                                          eta=opt.ddim_eta,
                                                          x_T=start_code)
                    x_samples_ddim = [decode_first_stage(model, samples_ddim[i:i+1].to(dtype=torch.float16))[0].cpu() for i in range(samples_ddim.size(0))]
                    x_samples_ddim = torch.stack(x_samples_ddim).float()
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    del samples_ddim
                    x_samples_ddim = accelerator.gather(x_samples_ddim) 

                    if accelerator.is_main_process and not opt.skip_save:
                        batch_images = []
                        for x_sample in x_samples_ddim:
                            x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                            x_sample = x_sample.astype(np.uint8)
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img.save(os.path.join(sample_path, f"original\\{base_count:08}_{seed_f}.png"))
                            if opt.resize_factor > 1.0:
                                img = upscale(img, opt.resize_factor)
                                img = np.array(img).astype(np.float32) / 255.0
                                img = np.moveaxis(img, 2, 0)
                                batch_images.append(img)
                        if opt.resize_factor > 1.0:
                            decoded_samples = torch.from_numpy(np.array(batch_images))
                            decoded_samples = decoded_samples.to(device)
                            decoded_samples = 2. * decoded_samples - 1.
                            samples = model.get_first_stage_encoding(model.encode_first_stage(decoded_samples))
                            samples = samples[:, :, 0//2:samples.shape[2]-1//2, 0//2:samples.shape[3]-1//2]
                            if type(opt.seed) == list:
                                all_seeds = opt.seed
                            else:
                                all_seeds = [int(opt.seed) + a for a in range(len(all_prompts))]
                            noise = create_random_tensors(samples.shape[1:], seeds=all_seeds, device=device)
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                            hr_steps, t_enc = setup_img2img_steps(opt, opt.hr_steps)
                            if k_d:
                                extra_params_kwargs = {}
                                if opt.sampler.endswith("_ka"):
                                    sigmas = K.sampling.get_sigmas_karras(hr_steps, sigma_min, sigma_max, device=device)
                                    opt.sampler = opt.sampler[:-3]
                                    karras = True
                                else:
                                    sigmas = model_wrap.get_sigmas(hr_steps)
                                if "dpm_2" in opt.sampler:
                                    sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
                                sigma_sched = sigmas[hr_steps - t_enc - 1:]
                                xi = samples + noise * sigma_sched[0]
                                def create_noise_sampler(x, sigmas, opt):
                                    from k_diffusion.sampling import BrownianTreeNoiseSampler
                                    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
                                    if type(opt.seed) == list:
                                        all_seeds = opt.seed
                                    else:
                                        all_seeds = [int(opt.seed) + a for a in range(len(all_prompts))]
                                    current_iter_seeds = all_seeds[n * opt.n_samples:(n + 1) * opt.n_samples]
                                    return BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=current_iter_seeds)
                                parameters = inspect.signature(K.sampling.__dict__[f'sample_{opt.sampler}']).parameters
                                if 'sigma_min' in parameters:
                                    ## last sigma is zero which isn't allowed by DPM Fast & Adaptive so taking value before last
                                    extra_params_kwargs['sigma_min'] = sigma_sched[-2]
                                if 'sigma_max' in parameters:
                                    extra_params_kwargs['sigma_max'] = sigma_sched[0]
                                if 'n' in parameters:
                                    extra_params_kwargs['n'] = len(sigma_sched) - 1
                                if 'sigma_sched' in parameters:
                                    extra_params_kwargs['sigma_sched'] = sigma_sched
                                if 'sigmas' in parameters:
                                    extra_params_kwargs['sigmas'] = sigma_sched
                                if opt.sampler == "dpmpp_sde":
                                    noise_sampler = create_noise_sampler(samples, sigmas, opt)
                                    extra_params_kwargs['noise_sampler'] = noise_sampler
                                samples_ddim = K.sampling.__dict__[f'sample_{opt.sampler}'](model_wrap_cfg, xi, extra_args=extra_args, disable=not accelerator.is_main_process, **extra_params_kwargs)
                                if karras:
                                    opt.sampler = opt.sampler + "_ka"
                                    karras = False
                            else:
                                x1 = sampler.stochastic_encode(samples, torch.tensor([t_enc] * int(samples.shape[0])).to(device), noise=noise)
                                samples_ddim = sampler.decode(x1, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,)
                            x_samples_ddim = [decode_first_stage(model, samples_ddim[i:i+1].to(dtype=torch.float16))[0].cpu() for i in range(samples_ddim.size(0))]
                            x_samples_ddim = torch.stack(x_samples_ddim).float()
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            del samples_ddim
                            x_samples_ddim = accelerator.gather(x_samples_ddim) 
                            base_count = 0
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                                x_sample = x_sample.astype(np.uint8)
                                img = Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"resized\\{base_count:08}_{seed_f}.png"))
                                base_count += 1

                    if accelerator.is_main_process and not opt.skip_grid:
                        all_samples.append(x_samples_ddim)

                if accelerator.is_main_process and not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()

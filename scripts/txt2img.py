import argparse, os, sys, glob, re
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

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

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
        
class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.step = 0
    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
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
        type=int,
        help="Resize factor",
        default=2
    )

    
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
                        extra_params_kwargs = {}
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
                            sigmas = K.sampling.get_sigmas_karras(opt.ddim_steps, sigma_min, sigma_max, device=device)
                            opt.sampler = opt.sampler[:-3]
                            karras = True
                        else:
                            sigmas = model_wrap.get_sigmas(opt.ddim_steps)
                        if "dpm_2" in opt.sampler:
                            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
                        torch.manual_seed(opt.seed) # changes manual seeding procedure
                        x = torch.randn([opt.n_samples, *shape], device=device) * sigmas[0] # for GPU draw
                        if "dpmpp_sde" in opt.sampler:
                            noise_sampler = create_noise_sampler(x, sigmas, opt)
                            extra_params_kwargs['noise_sampler'] = noise_sampler
                        # x = torch.randn([opt.n_samples, *shape]).to(device) * sigmas[0] # for CPU draw
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {'cond': c,  'uncond': uc, 'cond_scale': opt.scale}
                        
                        samples_ddim = K.sampling.__dict__[f'sample_{opt.sampler}'](model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, **extra_params_kwargs)
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
                        for x_sample in x_samples_ddim:
                            x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                            x_sample = x_sample.astype(np.uint8)
                            img = Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"original\\{base_count:08}_{seed_f}.png"))
                            resize_image(os.path.join(sample_path, f"original\\{base_count:08}_{seed_f}.png")
                                          , os.path.join(sample_path, f"resized\\{base_count:08}_{seed_f}.png")
                                          , opt.W, opt.H, opt.resize_factor)
                            improve_image(os.path.join(sample_path, f"resized\\{base_count:08}_{seed_f}.png")
                                          , os.path.join(sample_path, f"improved\\{base_count:08}_{seed_f}.png"))
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

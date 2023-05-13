import os
import re

import torch
from PIL import Image
import numpy as np

from scripts import deepbooru_model
from basicsr.utils.download_util import load_file_from_url

re_special = re.compile(r'([\\()])')


class DeepDanbooru:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is not None:
            return
        
        dl = load_file_from_url(
            model_url="https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt",
            model_path=os.getcwd(), 
            True, 
            download_name='model-resnet_custom_v3.pt',
        )

        self.model = deepbooru_model.DeepDanbooruModel()
        self.model.load_state_dict(torch.load(dl, map_location="cpu"))

        self.model.eval()
        self.model.to(torch.device("cpu"), torch.float16)

    def start(self):
        self.load()
        self.model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def stop(self):
        self.model.to(torch.device("cpu"))
        torch_gc()

    def tag(self, pil_image):
        self.start()
        res = self.tag_multi(pil_image)
        self.stop()

        return res

    def tag_multi(self, pil_image, force_disable_ranks=False):
        threshold = 0.5
        use_spaces = False
        use_escape = True
        alpha_sort = True
        include_ranks = not force_disable_ranks

        im = pil_image.convert("RGB")
        ratio = 1
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))
        pic = res
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

        with torch.no_grad(), torch.autocast("cuda"):
            x = torch.from_numpy(a).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            y = self.model(x)[0].detach().cpu().numpy()

        probability_dict = {}

        for tag, probability in zip(self.model.tags, y):
            if probability < threshold:
                continue

            if tag.startswith("rating:"):
                continue

            probability_dict[tag] = probability

        if alpha_sort:
            tags = sorted(probability_dict)
        else:
            tags = [tag for tag, _ in sorted(probability_dict.items(), key=lambda x: -x[1])]

        res = []

        for tag in [x for x in tags]:
            probability = probability_dict[tag]
            tag_outformat = tag
            if use_spaces:
                tag_outformat = tag_outformat.replace('_', ' ')
            if use_escape:
                tag_outformat = re.sub(re_special, r'\\\1', tag_outformat)
            if include_ranks:
                tag_outformat = f"({tag_outformat}:{probability:.3f})"

            res.append(tag_outformat)

        return ", ".join(res)


model = DeepDanbooru()

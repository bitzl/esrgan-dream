from dataclasses import dataclass
from pathlib import Path
import random
from typing import TextIO
from basicsr.archs.rrdbnet_arch import RRDBNet
import numpy as np
from realesrgan import RealESRGANer
import torch

import cv2

from esrgan_dream import ColorMode, IDGenerator, experiment_id
import yaml

ID_GENERATOR = IDGenerator()

def create_upsampler(model_path: str, tile: str):
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        return RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=True,
            gpu_id=None,
        )

class Dream:
    def __init__(
        self,
        numpy_seed: int,
        torch_seed: int,
        initial_width: int,
        initial_height: int,
        tile: int,
        color_mode: ColorMode,
        model_path: str,
        blur: int,
        color_offset: int,
    ):
        self.id = ID_GENERATOR.next()
        self.initial_width = initial_width
        self.initial_height = initial_height
        self.tile = tile
        self.color_mode = color_mode
        self.blur = blur
        self.model_path = model_path
        self.color_offset = color_offset

        if numpy_seed is None:
            numpy_seed = random.randint(0, 2 ** 32 - 1)
        self.numpy_seed = numpy_seed

        if torch_seed is not None:
            torch.manual_seed(torch_seed)
            self.torch_seed = torch_seed
        else:
            self.torch_seed = torch.seed()

    def dream(self, repeat_upscale, out: str, progress_callback: lambda: None):
        # Setup model. Don't do this in __init__ because this uses GPU memory.
        upsampler = create_upsampler(self.model_path, self.tile)
        img = self.blurry_noise()
        for iteration in range(
            repeat_upscale + 1
        ):  # add one because in the first iteration we just save the original image
            if iteration > 0:
                img, _ = upsampler.enhance(img, outscale=4)
            cv2.imwrite(
                f"{out}/{self.id}_{iteration:02d}.png", img
            )
            progress_callback()

    def blurry_noise(self):
        rng = np.random.default_rng(self.numpy_seed)
        max_value = min(255 - self.color_offset, 255) # allow negative offset to make colors darker
        width, height = self.initial_width, self.initial_height
        match self.color_mode:
            case ColorMode.color:
                print("Generate grayscale image")
                img = rng.integers(max_value, size=(height, width, 3), dtype=np.uint8) + self.color_offset
                if self.blur > 0:
                    img = cv2.blur(img, (self.blur, self.blur))
            case ColorMode.grayscale:
                print("Generate grayscale image")
                img = rng.integers(max_value, size=(height, width, 1), dtype=np.uint8) + self.color_offset
                if self.blur > 0:
                    img = cv2.blur(img, (self.blur, self.blur))
            case ColorMode.black_and_white:
                print("Generate black and white image")
                img = rng.integers(255, size=(height, width, 1), dtype=np.uint8)
                img[img < 128] = 0
                img[img >= 128] = 255
        return img

    def dumps(self) -> str:
        state = {
            "initial": {
                "width": self.initial_width,
                "height": self.initial_height,
                "color_mode": repr(self.color_mode),
                "blur": self.blur,
                "color_offset": self.color_offset,
            },
            "seeds": {
                "numpy": self.numpy_seed,
                "torch": self.torch_seed,
            },
            "tile": self.tile,
            "model_path": self.model_path
        }
        return yaml.dump(state)

    def dump(self, fp) -> str:
        fp.write(self.dumps())

    @classmethod
    def load(cls, fp: TextIO) -> "Dream":
        doc = yaml.safe_load(fp)

        initial = doc["initial"]
        seeds = doc["seeds"]

        return cls(
            numpy_seed=seeds["numpy"],
            torch_seed=seeds["torch"],
            initial_width=initial["width"],
            initial_height=initial["height"],
            tile=doc["tile"],
            color_mode=initial["color_mode"],
            model_path=doc["model_path"],
            blur=initial["blur"],
            color_offset=initial.get("color_offset", 0),
        )

# @dataclass
# class Seeds:
#     numpy: int
#     torch: int

class DreamFromImage:
    def __init__(
        self,
        image_path: Path,
        numpy_seed: int,
        torch_seed: int,
        tile: int,
        model_path: str,
    ):
        self.id = ID_GENERATOR.next()
        self.image_path = image_path
        self.tile = tile
        self.model_path = model_path

        if numpy_seed is None:
            self.numpy_seed = random.randint(0, 2 ** 32 - 1)
    
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
        self.torch_seed = torch_seed

    def dream(self, repeat_upscale, out: str, progress_callback: lambda: None):
        # Setup model. Don't do this in __init__ because this uses GPU memory.
        upsampler = create_upsampler(self.model_path, self.tile)

        img = cv2.imread(str(self.image_path))
        for iteration in range(
            repeat_upscale + 1
        ):  # add one because in the first iteration we just save the original image
            if iteration > 0:
                img, _ = upsampler.enhance(img, outscale=4)
            cv2.imwrite(
                f"{out}/{self.id}_{self.tile}_{iteration:02d}.png", img
            )
            progress_callback()

    def dumps(self) -> str:
        state = {
            "initial": {
                "width": self.initial_width,
                "height": self.initial_height,
                "image_path": self.image_path,
            },
            "seeds": {
                "numpy": self.numpy_seed,
                "torch": self.torch_seed,
            },
            "tile": self.tile,
            "model_path": self.model_path
        }
        return yaml.dump(state)

    def dump(self, fp) -> str:
        fp.write(self.dumps())

    @classmethod
    def load(cls, fp: TextIO) -> "Dream":
        doc = yaml.safe_load(fp)

        initial = doc["initial"]
        seeds = doc["seeds"]

        return cls(
            numpy_seed=seeds["numpy"],
            torch_seed=seeds["torch"],
            initial_width=initial["width"],
            initial_height=initial["height"],
            tile=doc["tile"],
            color_mode=initial["color_mode"],
            model_path=doc["model_path"],
            blur=initial["blur"],
            color_offset=initial.get("color_offset", 0),
        )
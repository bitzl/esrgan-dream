from enum import Enum
import math
import random
import cv2

import numpy as np

from esrgan_dream import ColorMode


class BlurType(str, Enum):
    mean = "mean"
    gaussian = "gaussian"


class BlurryNoiseGenerator:
    def __init__(
        self,
        width: int,
        height: int,
        blur_type: BlurType,
        color_mode: ColorMode,
        random_seed=None,
        blur_kernel_size=5,
        blur_sigma=1.0,
        color_offset=0,
        tile_size=None,
    ):
        self.width = width
        self.height = height
        self.blur_type = blur_type
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.tile_size = tile_size
        if random_seed is None:
            random_seed = random.randint(0, 2**32 - 1)
        self.random_seed = random_seed
        self.color_offset = color_offset
        self.color_mode = color_mode

    def __call__(self):
        rng = np.random.default_rng(self.random_seed)
        max_value = min(
            255 - self.color_offset, 255
        )  # allow negative offset to make colors darker
        print(f"Generate {self.color_mode} image")
        match self.color_mode:
            case ColorMode.color:
                img = (
                    rng.integers(
                        max_value, size=(self.height, self.width, 3), dtype=np.uint8
                    )
                    + self.color_offset
                )
                img = self._make_tiled(img)
                img = self._blur(img)
            case ColorMode.grayscale:
                img = (
                    rng.integers(
                        max_value, size=(self.height, self.width, 1), dtype=np.uint8
                    )
                    + self.color_offset
                )
                img = self._make_tiled(img)
                img = self._blur(img)
            case ColorMode.black_and_white:
                img = rng.integers(
                    255, size=(self.height, self.width, 1), dtype=np.uint8
                )
                img = self._make_tiled(img)
                img[img < 128] = 0
                img[img >= 128] = 255
                # no blurring, because it's black and white, which means no gray values as blurring would introduce
            case _:
                raise ValueError(f"Unknown color mode {self.color_mode}")
        return img

    def _blur(self, image):
        if self.blur_kernel_size == 0:
            return np.squeeze(image) # make sure the shape is the same as if blurring was applied
        if self.blur_type == BlurType.mean:
            return cv2.blur(image, (self.blur_kernel_size, self.blur_kernel_size))
        elif self.blur_type == BlurType.gaussian:
            return cv2.GaussianBlur(
                image, (self.blur_kernel_size, self.blur_kernel_size), self.blur_sigma
            )

    def _make_tiled(self, image):
        if self.tile_size is None or self.tile_size == 0:
            return image
        tile = image[0 : self.tile_size, 0 : self.tile_size]
        tile_shape = list(image.shape)
        tile_shape[0] = image.shape[0] // self.tile_size
        tile_shape[1] = image.shape[1] // self.tile_size
        return np.tile(tile, tile_shape)

    def state(self):
        return {
            "width": self.width,
            "height": self.height,
            "color_mode": self.color_mode.value,
            "blur": self.blur_type.value,
            "blur_kernel_size": self.blur_kernel_size,
            "blur_sigma": self.blur_sigma,
            "color_offset": self.color_offset,
            "seed": self.random_seed,
            "tile_size": self.tile_size,
        }

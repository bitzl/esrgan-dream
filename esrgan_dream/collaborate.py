from dataclasses import dataclass
from pathlib import Path
import random
from typing import List, Tuple
from PIL import Image
import cv2
import numpy as np
from esrgan_dream import ColorMode

from esrgan_dream.dream import create_upsampler
from esrgan_dream.source import BlurType, BlurryNoiseGenerator


@dataclass
class Params:
    repeat_upscale: int
    variations: int  # how many variation to generate
    pass


@dataclass
class GeneratedImage:
    input: np.ndarray
    blurred: np.ndarray
    generated: Image.Image


class Model:
    def __init__(self, params: Params) -> None:
        self.params = params
        self.original: GeneratedImage = None
        self.variations: List[GeneratedImage] = []
        self.upsampler = create_upsampler("weights/RealESRGAN_x4plus.pth", 512)
        self.iterations = 0
        self.reset()

    def reset(self) -> None:
        self.iterations = 0
        # Generate original image
        input = np.random.random((16, 16)).astype(np.float32) * 255
        blurred = cv2.blur(input, (3, 3))
        generated = self.upscale(blurred)
        self.original = GeneratedImage(input, blurred, generated)
        # Generate variations
        self.variations = []
        # self.variations = [
        #     self.vary(self.original) for _ in range(self.params.variations)
        # ]

    def iterate(
        self,
        selected: int,
        number_of_pixels_to_vary: int,
        max_variation_per_pixel: float,
    ) -> List[Image.Image]:
        self.iterations += 1
        self.original = self.variations[selected]
        self.variations = [
            self.vary(self.original, number_of_pixels_to_vary, max_variation_per_pixel)
            for _ in range(self.params.variations)
        ]
        return self.variations

    def update_variations(
        self, percentage_of_pixels_to_vary: float, max_variation_per_pixel: float
    ):
        """
        Just update the variations, but keep the original image the same.
        """
        self.variations = [
            self.vary(self.original, percentage_of_pixels_to_vary, max_variation_per_pixel)
            for _ in range(self.params.variations)
        ]

    def vary(
        self,
        image: GeneratedImage,
        percentage_of_pixels_to_vary: float,
        max_variation_per_pixel: float,
    ) -> GeneratedImage:
        # vary input
        input_variation = np.copy(image.input)
        all_coordinates = [
            (y, x)
            for y in range(input_variation.shape[0])
            for x in range(input_variation.shape[1])
        ]
        n = int(len(all_coordinates) * percentage_of_pixels_to_vary)
        print(n, len(all_coordinates), percentage_of_pixels_to_vary)
        coordinates = random.sample(
            all_coordinates, int(len(all_coordinates) * percentage_of_pixels_to_vary)
        )
        for x, y in coordinates:
            change = (
               255 *  max_variation_per_pixel * 2 * (random.random() - 0.5)
            )  # apply bounds symmetrically
            input_variation[y, x] += change
        input_variation.clip(0, 255)
        # create upscaled version that introduces acutal features
        blurred = cv2.blur(input_variation, (3, 3))
        image = self.upscale(blurred)
        return GeneratedImage(input_variation, blurred, image)

    def upscale(self, image: np.ndarray) -> np.ndarray:
        for _ in range(self.params.repeat_upscale):
            image, _ = self.upsampler.enhance(image, outscale=4)
        return Image.fromarray(np.uint8(image), "L")

    def save_original_to(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / f"{self.iterations:05d}.png"
        self.original.generated.save(file_path)

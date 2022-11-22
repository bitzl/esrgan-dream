from pathlib import Path
from random import random
import cv2
from pyparsing import TextIO
import torch
import yaml
from esrgan_dream import ColorMode
from esrgan_dream.dream import ID_GENERATOR, create_upsampler
from esrgan_dream.source import BlurryNoiseGenerator
from skimage.metrics import structural_similarity
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Inception:
    def __init__(
        self,
        torch_seed: int,
        tile: int,
        bng: BlurryNoiseGenerator,
        model_path: str,
        comment: str = None,
        grow_to: int = 4096,
    ):
        self.id = ID_GENERATOR.next()
        self.tile = tile
        self.model_path = model_path
        self.comment = comment
        self.bng: BlurryNoiseGenerator = bng
        self.structural_similarity = []
        self.grow_to = grow_to

        if torch_seed is not None:
            torch.manual_seed(torch_seed)
            self.torch_seed = torch_seed
        else:
            self.torch_seed = torch.seed()

    def dream(self, repeat: int, out: Path, progress_callback: lambda: None):
        # Setup model. Don't do this in __init__ because this uses GPU memory.
        upsampler = create_upsampler(self.model_path, self.tile)
        img = self.bng()
        upsampler.scale
        scale = 2
        for iteration in range(
            repeat + 1
        ):  # add one because in the first iteration we just save the original image
            if iteration > 0:
                tmp = img

                if max([img.shape[0], img.shape[1]]) < self.grow_to:
                    img, _ = upsampler.enhance(img, outscale=scale)
                    if self.bng.color_mode == ColorMode.color:
                        similarity = structural_similarity(tmp, cv2.resize(img, tmp.shape[:2]), channel_axis=2)
                    else:
                        similarity = structural_similarity(tmp, cv2.resize(img, tmp.shape[:2]))
                else:
                    img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
                    img, _ = upsampler.enhance(img, outscale=scale)
                    if self.bng.color_mode == ColorMode.color:
                        similarity = structural_similarity(tmp, img, channel_axis=2)
                    else:
                        similarity = structural_similarity(tmp, img)

                # if iteration % growth_cycles == 0:
                #     img = cv2.resize(img, (self.bng.width, self.bng.height))
                #     img, _ = upsampler.enhance(img, outscale=4)
                #     if self.bng.color_mode == ColorMode.color:
                #         similarity = structural_similarity(cv2.resize(tmp, img.shape), img, channel_axis=2)
                #     else:
                #         similarity = structural_similarity(cv2.resize(tmp, img.shape), img)
                # else:
                #     img, _ = upsampler.enhance(img, outscale=4)
                #     if self.bng.color_mode == ColorMode.color:
                #         similarity = structural_similarity(tmp, cv2.resize(img, tmp.shape), channel_axis=2)
                #     else:
                #         similarity = structural_similarity(tmp, cv2.resize(img, tmp.shape))
            else:
                similarity = 0
            self.structural_similarity.append(similarity)
            cv2.imwrite(
                f"{out}/{self.id}_{iteration:03d}.png", img
            )
            with (out / f"{self.id}.yml").open("w") as fp:
                self.dump(fp)
            self.save_metrics(out)
            progress_callback()

    def save_metrics(self, out: Path):
        with open(f"{out}/{self.id}_metrics.txt", "wt") as f:
            f.write("\n".join(f"{i}, {s}" for i, s in enumerate(self.structural_similarity)))
        plt.plot(self.structural_similarity)
        plt.xlabel("Iterations")
        plt.ylabel("Structural Similarity")
        plt.savefig(out / f"{self.id}_metrics.png")
        plt.close()

    def dumps(self) -> str:
        state = {
            "bng": self.bng.state(),
            "seeds": {
                "torch": self.torch_seed,
            },
            "grow_to": self.grow_to,
            "tile": self.tile,
            "model_path": self.model_path,
            "structural_similarity": self.structural_similarity,
            "iterations": len(self.structural_similarity)
        }
        if self.comment is not None:
            state["comment"] = self.comment
        return yaml.dump(state)

    def dump(self, fp) -> str:
        fp.write(self.dumps())

    @classmethod
    def load(cls, fp: TextIO) -> "Inception":
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
            comment=doc.get("comment", None)
        )

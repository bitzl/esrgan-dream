import random
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import os
import cv2
import numpy as np
import torch
from tqdm import trange
from pathlib import Path
from rich.progress import Progress
import typer
from yaml import CBaseDumper


def blurry_noise(width: int, height: int, channels: int = 3) -> np.ndarray:
    img = np.random.randint(255, size=(width, height, channels), dtype=np.uint8)
    return cv2.blur(img, (3, 3))


app = typer.Typer()


@app.command()
def main(
    model_path: str = typer.Option(
        "weights/RealESRGAN_x4plus.pth", help="Path to the model file"
    ),
    out: str = typer.Option("out", help="Path to the output folder"),
    repeat_upscale: int = typer.Option(3, help="Number of times to upscale the image"),
    experiments: int = typer.Option(1, help="Number of times to run the experiment"),
    channels: int = typer.Option(3, help="Number of color channels in the image"),
):
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
    )
    netscale = 4

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=None,
    )

    initial_width = 16
    Path("out").mkdir(parents=True, exist_ok=True)
    experiment_labels = [f"{random.randint(0, 99999):05d}" for _ in range(experiments)]

    with Progress() as progress:
        experiment_progress = [
            (
                label,
                progress.add_task(f"Experiment {i} ({label})", total=repeat_upscale),
            )
            for i, label in enumerate(experiment_labels)
        ]
        for label, task in experiment_progress:
            img = blurry_noise(initial_width, initial_width, channels)
            for iteration in range(
                repeat_upscale + 1
            ):  # add one because in the first iteration we just save the original image
                if iteration > 0:
                    img, _ = upsampler.enhance(img, outscale=4)
                cv2.imwrite(f"{out}/{label}_{iteration:02d}.png", img)
                progress.update(task, advance=1)


if __name__ == "__main__":
    typer.run(main)

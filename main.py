from enum import Enum
import random
import time
from typing import List, Tuple
import numpy as np
from tqdm import trange
from pathlib import Path
from rich.progress import Progress
import typer
from esrgan_dream import ColorMode
from esrgan_dream import dream

from esrgan_dream.dream import Dream, DreamFromImage
from esrgan_dream.inception import Inception
from esrgan_dream.source import BlurType, BlurryNoiseGenerator

MAX_SEED = 2**32 - 1

app = typer.Typer()


@app.command()
def run(
    experiment: Path = typer.Argument(..., help="Path to experiment file"),
    iterations: int = typer.Option(4, help="Number of upscaling iterations to run"),
    out: Path = typer.Option(Path("out"), help="Path to the output folder"),
):
    out.mkdir(exist_ok=True, parents=True)
    with experiment.open() as fp:
        dream = Dream.load(fp)
    with open(f"{out}/{dream.encoded_seed}.yml", "w") as fp:
        dream.dump(fp)
    with Progress() as progress:
        task = progress.add_task(
            f"Upscaling {experiment.name} (as {out}/{dream.encoded_seed}.yml)",
            total=iterations,
        )
        dream.dream(iterations, out, lambda: progress.advance(task))


@app.command()
def experiments(
    model_path: str = typer.Option(
        "weights/RealESRGAN_x4plus.pth", help="Path to the model file"
    ),
    out: Path = typer.Option(Path("out"), help="Path to the output folder"),
    iterations: int = typer.Option(3, help="Number of times to upscale the image"),
    experiments: int = typer.Option(1, help="Number of times to run the experiment"),
    color_mode: ColorMode = typer.Option(
        ColorMode.color, help="Number of color channels in the image"
    ),
    tile: int = typer.Option(512, help="Size for image tiles (0: no tiling)"),
    blur: int = typer.Option(3, help="Blur kernel size"),
    blur_type: BlurType = typer.Option(BlurType.mean, help="Blur type"),
    color_offset: int = typer.Option(
        0, help="Offset to make the image brighter or darker"
    ),
    initial_width: int = typer.Option(16, help="Initial width of the image"),
    initial_height: int = typer.Option(16, help="Initial height of the image"),
    comment: str = typer.Option(None, help="Comment to add to the output folder name"),
    input_tile: int = typer.Option(
        None, help="Create random input by repeating tiles of this size before blurring"
    ),
):
    out.mkdir(exist_ok=True, parents=True)
    numpy_seeds = [random.randint(0, MAX_SEED) for _ in range(experiments)]
    torch_seeds = [random.randint(0, MAX_SEED) for _ in range(experiments)]

    # setup experiments
    experiments = [
        Dream(
            torch_seed,
            tile,
            model_path,
            comment=comment,
            bng=BlurryNoiseGenerator(
                initial_width,
                initial_height,
                blur_type,
                color_mode,
                numpy_seed,
                blur,
                color_offset=color_offset,
                tile_size=input_tile,
            ),
        )
        for numpy_seed, torch_seed in zip(numpy_seeds, torch_seeds)
    ]

    start = time.time()
    with Progress() as progress:
        # setup progress tracking
        experiment_progress = [
            (
                experiment,
                progress.add_task(
                    f"Experiment {i} ({experiment.id})",
                    total=iterations,
                ),
            )
            for i, experiment in enumerate(experiments)
        ]
        # perform experiments
        for experiment, task in experiment_progress:
            with open(f"{out}/{experiment.id}.yml", "w") as fp:
                experiment.dump(fp)
            experiment.dream(iterations, out, lambda: progress.advance(task))
    stop = time.time()
    n = len(experiments)
    typer.secho(
        f"Finished {n} experiments in {stop - start:.2f} seconds ({(stop - start)/n:.2f} s/experiment))",
        fg=typer.colors.GREEN,
    )


@app.command()
def from_image(
    image_path: Path,
    out: Path = typer.Option(Path("out"), help="Path to the output folder"),
    model_path: str = typer.Option(
        "weights/RealESRGAN_x4plus.pth", help="Path to the model file"
    ),
    iterations: int = typer.Option(3, help="Number of times to upscale the image"),
    tile: int = typer.Option(512, help="Size for image tiles (0: no tiling)"),
):
    out.mkdir(exist_ok=True, parents=True)
    numpy_seed = random.randint(0, MAX_SEED)
    torch_seed = random.randint(0, MAX_SEED)

    if image_path.is_file():
        dreams = [DreamFromImage(image_path, numpy_seed, torch_seed, tile, model_path)]
    else:
        dreams = [
            DreamFromImage(image, numpy_seed, torch_seed, tile, model_path)
            for image in image_path.glob("*.png")
        ]

    with Progress() as progress:

        def tracking_task(dream):
            return progress.add_task(
                f"{dream.id}: Upscaling {dream.image_path.name}",
                total=iterations,
            )

        overall_progress = progress.add_task(
            "[medium_purple1]Total progress", total=len(dreams) * iterations
        )
        tasks = [(dream, tracking_task(dream)) for dream in dreams]
        for dream, task in tasks:

            def update():
                progress.advance(overall_progress)
                progress.advance(task)

            dream.dream(iterations, out, update)


@app.command()
def inception(
    width: int = typer.Argument(1024, help="Initial width of the image"),
    height: int = typer.Argument(1024, help="Initial height of the image"),
    grow_to: int = typer.Option(2048, help="Final width and height of the image"),
    model_path: str = typer.Option(
        "weights/RealESRGAN_x4plus.pth", help="Path to the model file"
    ),
    out: Path = typer.Option(Path("out"), help="Path to the output folder"),
    iterations: int = typer.Option(5, help="Number of times to upscale the image"),
    experiments: int = typer.Option(1, help="Number of times to run the experiment"),
    color_mode: ColorMode = typer.Option(
        ColorMode.color, help="Number of color channels in the image"
    ),
    tile: int = typer.Option(512, help="Size for image tiles (0: no tiling)"),
    blur: int = typer.Option(3, help="Blur kernel size"),
    blur_type: BlurType = typer.Option(BlurType.mean, help="Blur type"),
    color_offset: int = typer.Option(
        0, help="Offset to make the image brighter or darker"
    ),

    comment: str = typer.Option(None, help="Comment to add to the output folder name"),
    input_tile: int = typer.Option(
        None, help="Create random input by repeating tiles of this size before blurring"
    ),
):
    out.mkdir(exist_ok=True, parents=True)

    # setup experiments
    actual_experiments = [
        Inception(
            torch_seed,
            tile,
            grow_to=grow_to,
            model_path=model_path,
            comment=comment,
            bng=BlurryNoiseGenerator(
                width,
                height,
                blur_type,
                color_mode,
                numpy_seed,
                blur,
                color_offset=color_offset,
                tile_size=input_tile,
            ),
        )
        for numpy_seed, torch_seed in generate_seeds(experiments)
    ]

    timer = ExperimentTimer()
    with Progress() as progress:
        # setup progress tracking
        experiment_progress = setup_progress(progress, actual_experiments, iterations)
        # perform experiments
        for experiment, task in experiment_progress:
            experiment.dream(iterations, out, lambda: progress.advance(task))
    timer.stopAndPrint(experiments)


def generate_seeds(experiments: List) -> Tuple[List[int], List[int]]:
    numpy_seeds = [random.randint(0, MAX_SEED) for _ in range(experiments)]
    torch_seeds = [random.randint(0, MAX_SEED) for _ in range(experiments)]
    return zip(numpy_seeds, torch_seeds)

def setup_progress(progress, experiments, iterations):
    return [
        (
            experiment,
            progress.add_task(
                f"Experiment {i} ({experiment.id})",
                total=iterations,
            ),
        )
        for i, experiment in enumerate(experiments)
    ]


class ExperimentTimer():
    def __init__(self) -> None:
        self.start = time.time()
    def stopAndPrint(self, experiments: int):
        stop = time.time()
        typer.secho(
            f"Finished {experiments} experiments in {stop - self.start:.2f} seconds ({(stop - self.start)/experiments:.2f} s/experiment))",
            fg=typer.colors.GREEN,
        )

if __name__ == "__main__":
    app()

from dataclasses import dataclass
from typing import Optional
from rich.progress import Progress
import mlflow
import numpy as np
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

@dataclass
class Dimensions:
    width: int
    height: int

@dataclass
class GenerationParams:
    initial_size: Dimensions
    growth_steps: int
    initial_blur: int
    tile_size: int

@dataclass
class VariationParams:
    pixels_to_vary: int
    variation_pct: float

@dataclass
class ClassificationParams:
    batch_size: int
    classifier_model: str

@dataclass
class Params:
    generation: GenerationParams
    variation: VariationParams
    classification: ClassificationParams
    iterations: int


class State:
    best_score: float = 0
    best_image: np.ndarray = None
    best_class: int = None

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

class Model:
    def __init__(self, params: Params) -> "Model":
        self.params = params
        self.upsampler = create_upsampler(params.classification.classifier_model, params.generation.tile_size)
        # TODO: Load classifier model
        # TODO: setup tracking

    def run(self):
        with Progress() as progress:
            task = progress.add_task("Evolve", total=self.params.iterations)
            self.state = self.initial_state()
            for iteration in range(self.params.iterations):
                previous_state = self.state # to report differences
                self.state = self.next(iteration)
                eta = progress.tasks[task].time_remaining()
                self.track_progress(iteration, previous_state, eta)
                progress.update(task, advance=1)

    def initial_state(self) -> State:
        params = self.params.generation
        image = np.random.random((params.initial_size.height, params.initial_size.width)) # only grayscale for now
        if params.initial_blur is not None and params.initial_blur > 0:
            image = cv2.blur(image, (params.initial_blur, params.initial_blur))
        
        # TODO: classify image

        return None

    def next(self, iteration: int):
        # TODO generate variations
        # TODO classify variations
        # TODO select best variation
        pass

    def track_progress(self, iteration: int, previous_state: State, eta: Optional[float]):
        if eta is not None:
            mlflow.log_metric("eta", eta)
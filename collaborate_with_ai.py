import base64
from io import BytesIO
from pathlib import Path
from esrgan_dream.collaborate import Model, Params
import streamlit as st
from st_clickable_images import clickable_images
from PIL import Image
import numpy as np
import cv2

OUT_DIR = Path("out/interactive")


DEFAULTS = {
    "max_pixel_variation": 0.1,
    "percentage_of_pixels_to_vary": 0.1,
}

def image_to_base64(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, "JPEG")
    image = output.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(image).decode()


def numpy_to_base64(image: np.ndarray) -> str:
    image = Image.fromarray(np.uint8(image * 255), "L")
    return image_to_base64(image)


if "model" not in st.session_state:
    st.session_state["model"] = Model(
        Params(
            repeat_upscale=2,
            variations=9,
        )
    )
    st.session_state["model"].update_variations(
        DEFAULTS["percentage_of_pixels_to_vary"], DEFAULTS["max_pixel_variation"]
    )
    st.session_state["model"].save_original_to(OUT_DIR)

model: Model = st.session_state["model"]  # assign to variable for readble code

col1, col2 = st.columns([1, 3])

with col1:
    image_container = st.empty()
    image_container.image(model.original.generated)

with col2:
    percentage_of_pixels_to_vary = st.slider(
        "Number of pixels to vary", 0.0, 1.0, value=DEFAULTS["percentage_of_pixels_to_vary"], step=0.01
    )
    max_variation_per_pixel = st.slider(
        "Max variation per pixel", 0.0, 1.0, value=DEFAULTS["max_pixel_variation"], step=0.01
    )

tab_generated, tab_inputs, tab_blurred = st.tabs(["Generated Variation", "Input", "Blurred Input"])
with tab_generated:
    variations = clickable_images(
        [image_to_base64(image.generated) for image in model.variations],
        img_style={"padding": "3px", "max-width": "33.3%"},
        key="variations-generated",
    )

with tab_inputs:
    input = clickable_images(
        [numpy_to_base64(cv2.resize(image.input, (256, 256), interpolation=cv2.INTER_NEAREST)) for image in model.variations],
        img_style={"padding": "3px", "max-width": "33.3%"},
        key="variations-input",
    )

with tab_blurred:
    blurred = clickable_images(
        [numpy_to_base64(cv2.resize(image.blurred, (256, 256), interpolation=cv2.INTER_NEAREST)) for image in model.variations],
        img_style={"padding": "3px", "max-width": "33.3%"},
        key="variations-blurred",
    )

selected_variation = max(variations, input, blurred)
if selected_variation > -1:
    model.save_original_to(OUT_DIR)
    model.iterate(variations, percentage_of_pixels_to_vary, max_variation_per_pixel)
    print("Iterated", model.iterations, "variations was", variations)
else:
    model.update_variations(percentage_of_pixels_to_vary, max_variation_per_pixel)

image_container.empty()
image_container.image(model.original.generated)

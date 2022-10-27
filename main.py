from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

import cv2
import numpy as np
from tqdm import trange

model_path = "weights/RealESRGAN_x4plus.pth"
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
netscale = 4


upsampler = RealESRGANer(
    scale=netscale,
    model_path=model_path,
    dni_weight=None,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    gpu_id=None)


initial_width = 16
img = np.random.randint(255, size=(initial_width,initial_width,3),dtype=np.uint8)
img = cv2.blur(img,(3,3))

for i in trange(4):
    if i > 0:
        img, _ = upsampler.enhance(img, outscale=4)
    cv2.imwrite(f"out/img_{i:02d}.png", img)

import torch
from PIL import Image
import numpy as np
import os 
import cv2 

from upscale import StableUpscaler



os.makedirs("outputs", exist_ok=True)
upscaler = StableUpscaler() 
files = os.listdir("faces")

for imagepath in files:
    face = cv2.imread('faces' + os.sep + imagepath)
    #upscale again using stable diffusion 2.0 4x upscaler (512x512 -> 2048x2048)
    prompt = "face"
    num_steps = 50
    face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)) #cv2 -> PIL
    face = upscaler.upscale(face , prompt, num_steps , noise_level = 25, guidance_scale= 0)
    upscaledFace = np.array(upscaledFace)[:, :, ::-1].copy()  #PIL -> cv2
    cv2.imwrite("outputs" + os.sep + imagepath, upscaledFace)


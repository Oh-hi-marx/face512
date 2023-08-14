import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
import os 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

files = os.listdir("inputs")
for imagepath in files:
    image = Image.open('inputs' + os.sep + imagepath).convert('RGB')

    sr_image = model.predict(image)

    sr_image.save("outputs" + os.sep + imagepath)

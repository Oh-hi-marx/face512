

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torchvision import models
import torch.optim as optim
import torch.nn as nn
import cv2
import wandb
import os
import glob
import re
import numpy as np
import random
from PIL import Image,ImageFilter
import re

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

class simpDataset(Dataset):
	def __init__(self, imgs, transforms):
		self.imgs = imgs
		self.transforms = transforms
	def __len__(self):
	    return (len(self.imgs))
	def __getitem__(self, i):
		return self.transforms(Image.fromarray(self.imgs[i]))
class qualityFace:
    def __init__(self, weights= "./qualitynet/checkpoints/0_100.0.pth", classN=2):
    	self.transforms = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
        ])
    	self.numClass = classN
    	self.device = 'cuda'
    	model = models.regnet_y_400mf(weights = "IMAGENET1K_V2")
    	num_ftrs = model.fc.in_features
    	model.fc = nn.Linear(num_ftrs, self.numClass)
    	model = nn.Sequential(model, nn.Sigmoid())

    	checkpoint = torch.load(weights)
    	model.load_state_dict(checkpoint['model_state_dict'])
    	self.model = model.to(self.device)
    	self.model.eval()

    def pred(self, imgs):
        dataset = simpDataset(imgs, self.transforms)
        dataloader =DataLoader(dataset, batch_size=20, shuffle=False,num_workers=0)
        preds = []
        with torch.no_grad():
            for batchi, inputs in enumerate(dataloader):
                inputs= inputs.to(self.device)
                outputs = self.model(inputs)
                preds+= (outputs.detach().cpu().tolist())
        return preds

if __name__ == "__main__":
	qualitynet = qualityFace(weights = './checkpoints/2_100.0.pth')
	imgs = glob.glob("faces/*")
	imgs = natural_sort(imgs)
	for imgp in imgs:
		img = cv2.imread(imgp)
		pred = qualitynet.pred([img])
		print(imgp,pred)

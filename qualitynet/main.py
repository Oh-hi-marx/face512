

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
from PIL import Image,ImageFilter,ImageEnhance
from io import StringIO

from utils import  *

def visulise(tensor,step, epoch, labels):
    os.makedirs('visualise', exist_ok=True)
    print(tensor[0].shape)
    for i in range(tensor.shape[0]):
        img = tensor[i].permute(1, 2, 0)  .cpu().detach().numpy()[:, :, ::-1]
        label = labels[i].cpu().detach()
        cv2.imwrite("visualise/"+str(step) +"_"+str(epoch) +"_" + str(i)+ "_label_"+ str(label.tolist())+".jpg", img*255)

class qualityFace(Dataset):
    def __init__(self, imgs, transforms,PILtransform):
        self.imgs= imgs
        self.transforms = transforms
        self.PILtransform = PILtransform
        #self.facecutouts = glob.glob("U-2-Net/test_data/facecutouts/*")
        #self.cutouts = glob.glob("U-2-Net/test_data/cutouts/*")
        #self.facecutlen = (len(self.facecutouts))
        #self.cutlen = (len(self.cutouts))
    def __len__(self):
        return (len(self.imgs))

    def __getitem__(self, i):
       img = self.imgs[i]
       img = Image.open(img)
      # label = [0,0,0,0,0, 0] #[lowres, blur,2 face, obscured, missing crop (black background), noDegraded]
       #degradations
       noDegraded = 1
       highres = 1
       qualityNumber = random.random()
       if(qualityNumber > 0.8): # resize
           highres = 0
           min =5
           maxx = 100
           size = random.randrange(min, maxx)
           img = img.resize((size, size))
           #label[0] = size
           highres = 0
           img = jpegCompression(img, random.randrange(1,15))
           img = blur(img, 0,10)
       elif(qualityNumber >0.6):
           min =100
           maxx = 250
           size = random.randrange(min, maxx)
           img = img.resize((size, size))
           highres = 0.25
           img = jpegCompression(img, random.randrange(15,35))
           img = blur(img, 0,3)
       elif(qualityNumber >0.4):
           min =250
           maxx = 400
           size = random.randrange(min, maxx)
           img = img.resize((size, size))
           highres = 0.5
           img = jpegCompression(img, random.randrange(35,60))
           img = blur(img, 0,2)
       elif(qualityNumber >0.2):
           min =400
           maxx = 512
           size = random.randrange(min, maxx)
           img = img.resize((size, size))
           highres = 0.75
           img = jpegCompression(img, random.randrange(60,100))
       else:
           min = 512
           maxx = 650
           size = random.randrange(min, maxx)
           img = img.resize((size, size))






     
   


       #label[5] = noDegraded
       label  = 1
       img = self.transforms(img)

       return img, torch.FloatTensor([label,highres])

if __name__ == "__main__":
    useWandb =1
    imgs = glob.glob("ffhq/*")[:]
    epochs = 100
    device = 'cuda'
    numClass = 2
    learning_rate = 0.0001
    batch_size = 100
    if(useWandb):
        wandb.init(project="facequality-regnet-512")


    transform = transforms.Compose([
        transforms.Resize(650),
        transforms.ColorJitter(brightness=(0.5,1.3),contrast=(0.4),saturation=(0.7,1.4),hue=(-0.1,0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.001, 0.85)),
        transforms.RandomAdjustSharpness(1.4, p=0.2),
        transforms.RandomAutocontrast(p=0.1),
        transforms.RandomRotation(10),
        transforms.RandomPerspective(distortion_scale=0.05, p=0.05),
        transforms.RandomResizedCrop(512, scale=(0.6, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),

    ])
    PILtransform = transforms.Compose([
            transforms.ColorJitter(brightness=(0.5,1.3),contrast=(0.3),saturation=(0.8,1.2),hue=(-0.2,0.2)),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.001, 0.3)),
            transforms.RandomAdjustSharpness(1.4, p=0.2),
            transforms.RandomAutocontrast(p=0.1),
            transforms.RandomRotation(180),
            transforms.RandomPerspective(distortion_scale=0.05, p=0.05)])

    dataset = qualityFace(imgs[:-100], transform,PILtransform)
    dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = 5)
    testdataset = qualityFace(imgs[-100:], transform,PILtransform)
    testloader = DataLoader(testdataset, batch_size = batch_size, num_workers = 5)

    model = models.regnet_y_400mf(weights = "IMAGENET1K_V2")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, numClass)
    model = nn.Sequential(model, nn.Sigmoid())
    if(1):
        checkpoint = torch.load("checkpoints/2_98.0.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs('checkpoints', exist_ok=True)

    step = 0
    for epoch in range(epochs):

        for batchi, (inputs, labels) in enumerate(dataloader):
            #visulise(inputs, batchi, epoch, labels)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # propagate the loss backward
            loss.backward()
            # update the gradients

            optimizer.step()

            if(step%10==0 ):
                loss = loss.item()
                print(loss)
                if(useWandb):
                    wandb.log({ 'train loss': loss}, step = step)
            step+= 1

        ############ TEST ###############
        with torch.no_grad():
            loss =0;correct = 0;total = 0

            for inputs, labels in testloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                total += labels.size(0)
                loss += criterion(outputs, labels)
                correct += torch.sum(torch.abs(outputs.data - labels.data) <0.5)
            accuracy = (   100 * correct / (total*numClass))
            print('Accuracy of the network on test images: %0.3f %%' % accuracy)
            if(useWandb):
                wandb.log({'test accuracy': accuracy, 'epoch': epoch, 'test loss': loss})


        torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
            'checkpoints' + os.sep + str(epoch)+'_'+ str(round(accuracy.item(),2)) + '.pth')

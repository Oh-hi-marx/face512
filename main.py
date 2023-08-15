import torch
from PIL import Image
import numpy as np
import os 
from os import listdir
from os.path import isfile, join
import cv2 

from utils import *
#codeformer pip fork https://github.com/Oh-hi-marx/codeformer-pip
from codeformer.app import inference_app 
from ultralytics import YOLO
import math
from upscale import StableUpscaler

class Face512:
    def __init__(self, STABLEDIFFUSION = True, BACKGROUNDSIZE = 6000):
        self.stable_diffusion = STABLEDIFFUSION
        self.num_steps = 50
        self.noise_level = 25

        self.backgroundsize = BACKGROUNDSIZE
        self.scale = 10

        self.detector = YOLO('yolov8n-face.pt') 
        self.upscaler = StableUpscaler() #stable diffusion 4x upscaler #https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale

    def run(self, path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        os.makedirs(path + os.sep + "outputs", exist_ok=True)
        for imagepath in files:
            image = cv2.imread(path + os.sep + imagepath)
            image = resizeSide(image, self.backgroundsize)

            h,w,c = image.shape
            #image = cv2.resize(image, (int(w*self.scale), int(h*self.scale)) )
            
            #detect and align faces
            results = self.detector(image, conf= 0.4, verbose=False)

            #restore each face
            restoredFaces = []

            boxes = results[0].boxes.cpu().numpy().xyxy  # Boxes object for bbox outputs
            keypoints = results[0].keypoints.cpu().numpy().xy  # Keypoints object for pose outputs
            boxesConf = results[0].boxes.cpu().numpy().conf
            h,w,c = image.shape
            blank_image = np.zeros((int(h*2),int(w*2),3), np.uint8)
            blank_image[int(h/2):int(h/2)+h, int(w/2):int(w/2)+w] = image #black padding to allow for expanded bboxes
            image = blank_image 
            for i in range(len(boxes)):
                ### Box ops ####
                box= list( boxes[i])
                boxConf = boxesConf[i]
                #add padding offset to bbox 
                box = [box[0]+int(w/2), box[1]+int(h/2), box[2]+int(w/2), box[3]+int(h/2)]
                #expand bbox
                box = expandBox(box, expansion = 0.6)
                #create sqaure aspect ratio 
                box = squareBox(box)
        
                #crop face using bounding box
                face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                
                boxWidth = box[2] - box[0]
                boxHeight = box[3] - box[1]


                #facial keypoints
                landmarks = list(keypoints[i])
                points = [] #face points relative to bounding box
                for j, l in enumerate(landmarks):
                    point = (int(l[0]-box[0]),int( l[1] - box[1]))
                    points.append(point)
                
                #find angle of eyes relative to horizon 
                eyeAngle = findEyeAngle(points)

                face = rotate_image(face, eyeAngle)
                #crop to remove black rotation boarders
                fh,fw,fc = face.shape
                cropRatio = 5
                faceCropped = face[int((0.8/cropRatio)*fh):fh-int((1.2/cropRatio)*fh), int((1/cropRatio)*fw):fw-int((1/cropRatio)*fw)]
                

                #limit codeformer to faces under 512x512, otherwise bad results due to codeformer only trained on low res faces
                faceLimit = 512
                if(faceCropped.shape[0]<faceLimit or faceCropped.shape[1] <faceLimit):
                    #restore face from low quality using codeformer (? -> 512x512)
                    restoredFace = inference_app(
                                    image=faceCropped,
                                    background_enhance=False,
                                    face_upsample=False,
                                    upscale=1,
                                    has_aligned=False,
                                    codeformer_fidelity=0.75,
                                    only_center_face=True
                                    )
                else:
                    restoredFace = faceCropped


                #upscaler further with stable diffusion 4x upscaler  (512x512 -> 2048x2048)
                if(self.stable_diffusion):
                    prompt = "face"           
                    restoredFace = cv2.resize(restoredFace, (512,512))
                    restoredFace = Image.fromarray(cv2.cvtColor(restoredFace, cv2.COLOR_BGR2RGB)) #cv2 -> PIL
                    upscaledFace = self.upscaler.upscale(restoredFace , prompt, self.num_steps , noise_level = self.noise_level, guidance_scale= 0)
                    restoredFace = np.array(upscaledFace)[:, :, ::-1].copy()#PIL -> cv2
                    
                cv2.imwrite(path+ os.sep + "outputs" + os.sep  +"face_" +imagepath, restoredFace)
                #Ensure square output
                restoredFace = squareImg(restoredFace)

                #We cropped the central portion of the face to avoid black boarders
                # now we need to paste it back, but the face has been upscaled 
                reh, rew, _ = restoredFace.shape
                faceRatio = cropRatio - 2
                oneOverCropRatio = restoredFace.shape[0]/faceRatio # 512/ 4 = 128
                fullSize = int(cropRatio * oneOverCropRatio) #6* 128 

                #resize original low res face so 512x512 can fit into the crop
                face = cv2.resize(face, (fullSize, fullSize))
                face[int((0.8/cropRatio )* fullSize): reh + int((0.8/cropRatio) * fullSize), int(1/cropRatio * fullSize):rew+ int(1/cropRatio *fullSize)] = restoredFace

                #we need to unrotate and paste back onto the original background, while removing black boarders from rotating
                restoredFace = unrotateAndPaste(face, image, eyeAngle, box)

                restoredFace = cv2.resize( restoredFace,(int(boxWidth), int(boxHeight)) ) 
                #paste restored face bounding box onto original image
                image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]  = restoredFace
            
            #remove padding
            image = image[int(h/2):int(h/2)+h, int(w/2):int(w/2)+w] 
            cv2.imwrite(path+ os.sep + "outputs" + os.sep  +imagepath, image)

#Face512 upscales faces in a folder. 
if __name__ == "__main__":
    path = 'inputs3'

    face512 = Face512(STABLEDIFFUSION = 1)
    face512.run(path)
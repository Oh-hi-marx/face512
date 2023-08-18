import torch
from PIL import Image
import numpy as np
import os 
from os import listdir
from os.path import isfile, join
import cv2 
from tqdm import tqdm
from natsort import natsorted

from utils import *
#codeformer pip fork https://github.com/Oh-hi-marx/codeformer-pip
from codeformer.app import inference_app 
from ultralytics import YOLO
import math
from upscale import StableUpscaler
current = os.getcwd()
os.chdir("LDSR")
from LDSR.LDSR import LDSR
os.chdir(current)


from qualitynet.qualitynet import qualityFace
class Face512:
    def __init__(self, STABLEDIFFUSIONSTEPS = 50, BACKGROUNDSIZE = 15000, upscale = 8):
        self.num_steps = STABLEDIFFUSIONSTEPS
        self.noise_level = 25
        self.upscale = upscale

        self.backgroundsize = BACKGROUNDSIZE
        self.scale = 10

        self.detector = YOLO('yolov8n-face.pt') 
        self.upscaler = StableUpscaler() #stable diffusion 4x upscaler #https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale

        self.qualitynet = qualityFace(weights = './qualitynet/checkpoints/2_100.0.pth')
        self.faceLimit = 0.94

        self.ldsr = LDSR('LDSR/model.ckpt', 'LDSR/project.yaml')
    def run(self, path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        files = natsorted(files)
        os.makedirs(path + os.sep + "outputs", exist_ok=True)
        for imagepath in tqdm(files):
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
                
                #cv2.imwrite(path+ os.sep + "outputs" + os.sep  +"face_" +imagepath, faceCropped)
                #limit codeformer to faces under 512x512, otherwise bad results due to codeformer only trained on low res faces
                faceQuality = self.qualitynet.pred([faceCropped])[0][1]

                if(faceQuality < self.faceLimit):
                    #restore face from low quality using codeformer (? -> 512x512)
                    faceCropped = cv2.resize(faceCropped, (450,450)) #this needs to be here or codeformer will not produce sharp results
                    restoredFace = inference_app(
                                    image=faceCropped,
                                    background_enhance=False,
                                    face_upsample=False,
                                    upscale=1,
                                    has_aligned=False,
                                    codeformer_fidelity=0.8,
                                    only_center_face=True
                                    )
                    #cv2.imwrite(path+ os.sep + "outputs" + os.sep  +"facecode_" +imagepath, restoredFace)
                else:
                    print("skipping codeformer, face already high quality")
                    restoredFace = faceCropped


                #upscaler further with stable diffusion 4x upscaler  (512x512 -> 2048x2048)
                if( self.num_steps and self.upscale >= 8):
                    prompt = "face"           
                    restoredFace = cv2.resize(restoredFace, (512,512)) 
                    restoredFace = Image.fromarray(cv2.cvtColor(restoredFace, cv2.COLOR_BGR2RGB)) #cv2 -> PIL
                    upscaledFace = self.upscaler.upscale(restoredFace , prompt, self.num_steps , noise_level = self.noise_level, guidance_scale= 0)
                    upscaledFace.save(path+ os.sep + "outputs" + os.sep  +"intermediate_" +imagepath)
                    if(self.upscale>=16):
                        w,h = upscaledFace.size
                        upscaledFace = upscaledFace.resize((int(w/2), int(h/2)))
                        current = os.getcwd()
                        os.chdir("LDSR")
                        upscaledFace = self.ldsr.superResolution(upscaledFace,ddimSteps = self.num_steps)
                        os.chdir(current)

                    restoredFace = np.array(upscaledFace)[:, :, ::-1].copy()#PIL -> cv2

                    #split 2048x2048 into quadrants for further upscalingr
                    '''
                    if(self.upscale>=16):
                        h,w,c = restoredFace.shape
                        upscaledFace =np.zeros((int(w*2),int(h*2),c), np.uint8) #created empty upscaled background
                        uh,uw,uc = upscaledFace.shape
                        for x in range(0,2):
                            for y in range(0,2):    
                                restoredFaceQuadrant = restoredFace[int(y*h/2):int((y+1)*h/2), int(x*w/2):int((x+1)*w/2)] #get quadrant
                                restoredFaceQuadrant = cv2.resize(restoredFaceQuadrant, (512,512)) #reduce size to reduce compute
                                restoredFaceQuadrant = Image.fromarray(cv2.cvtColor(restoredFaceQuadrant, cv2.COLOR_BGR2RGB)) #cv2 -> PIL
                                restoredFaceQuadrant = self.upscaler.upscale(restoredFaceQuadrant , prompt, self.num_steps , noise_level = self.noise_level, guidance_scale= 0)
                                restoredFaceQuadrant = np.array(restoredFaceQuadrant)[:, :, ::-1].copy()
                                upscaledFace[int(y*uh/2):int((y+1)*uh/2), int(x*uw/2):int((x+1)*uw/2)] = restoredFaceQuadrant #put quadrant into upscaled background
                        restoredFace = upscaledFace 
                    '''    
                    #cv2.imwrite(path+ os.sep + "outputs" + os.sep  +"facestb_" +imagepath, restoredFace)
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
                cv2.imwrite(path+ os.sep + "outputs" + os.sep  +"facefinalrr_" +imagepath, restoredFace)
            #remove padding
            image = image[int(h/2):int(h/2)+h, int(w/2):int(w/2)+w] 
            cv2.imwrite(path+ os.sep + "outputs" + os.sep  +imagepath, image)

#Face512 upscales faces in a folder. 
if __name__ == "__main__":
    path = 'inputs6'
    STABLEDIFFUSIONSTEPS =100
    upscale = 16
    face512 = Face512(STABLEDIFFUSIONSTEPS = STABLEDIFFUSIONSTEPS, upscale = upscale)
    face512.run(path)
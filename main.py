import torch
from PIL import Image
import numpy as np
import os 
import cv2 

#codeformer pip fork https://github.com/Oh-hi-marx/codeformer-pip
from codeformer.app import inference_app 
from ultralytics import YOLO
import math
from upscale import StableUpscaler

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

detector = YOLO('yolov8n-face.pt') 
#upscaler = StableUpscaler() #stable diffusion 4x upscaler #https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale

files = os.listdir("inputs")
os.makedirs("outputs", exist_ok=True)
for imagepath in files:
    image = cv2.imread('inputs' + os.sep + imagepath)
    print(imagepath)
    #detect and align faces
    results = detector(image, conf= 0.4, verbose=False)


    #restore each face
    restoredFaces = []


    boxes = results[0].boxes.cpu().numpy().xyxy  # Boxes object for bbox outputs
    keypoints = results[0].keypoints.cpu().numpy().xy  # Keypoints object for pose outputs
    boxesConf = results[0].boxes.cpu().numpy().conf
    for i in range(len(boxes)):
        box= list( boxes[i])
        landmarks = list(keypoints[i])
        boxConf = boxesConf[i]
        print(boxConf, landmarks)
        face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

        points = [] #face points relative to bounding box
        for i, l in enumerate(landmarks):
            point = (int(l[0]-box[0]),int( l[1] - box[1]))
            
            points.append(point)
        
        lefteye= points[0]
        righteye = points[1]
        nose = points[2]
        #face = cv2.circle(face, nose, radius=1, color=(0, int(255), 0), thickness=-1)


 
        cv2.imwrite("outputs" + os.sep + str(i) +imagepath, face)

'''

        #restore face from low quality using codeformer (? -> 512x512)
        restoredFace = inference_app(
                        image=face,
                        background_enhance=False,
                        face_upsample=False,
                        upscale=2,
                        has_aligned=False,
                        codeformer_fidelity=0.75,
                        only_center_face=True
                        )
        cv2.imwrite("outputs" + os.sep + str(i) +imagepath, restoredFace)
    
        #upscale again using stable diffusion 2.0 4x upscaler (512x512 -> 2048x2048)
        prompt = "face"
        num_steps = 50

        restoredFace = cv2.resize(restoredFace, (512,512))
        restoredFace = Image.fromarray(cv2.cvtColor(restoredFace, cv2.COLOR_BGR2RGB)) #cv2 -> PIL
        upscaledFace = upscaler.upscale(restoredFace , prompt, num_steps , noise_level = 25, guidance_scale= 0)
        upscaledFace = np.array(upscaledFace)[:, :, ::-1].copy()#PIL -> cv2
        restoredFaces.append(upscaledFace)  

    for restoredFace in restoredFaces:
        cv2.imwrite("outputs" + os.sep + imagepath, restoredFace)
    '''
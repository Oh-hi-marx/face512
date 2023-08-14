import torch
from PIL import Image
import numpy as np
import os 
import cv2 

from codeformer.app import inference_app
from facelib import FaceDetector

from upscale import StableUpscaler


detector = FaceDetector()
#upscaler = StableUpscaler() 

files = os.listdir("inputs")
for imagepath in files:
    image = cv2.imread('inputs' + os.sep + imagepath)

    #detect and align faces
    faces, boxes, scores, landmarks = detector.detect_align(image)

    #restore each face
    restoredFaces = []

    for i in range(len(boxes)):
        box = boxes[i].cpu().numpy()
        landmark = landmarks[i].cpu().numpy()
        print(box, landmark)
        face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        cv2.imwrite("outputs" + os.sep + imagepath, face)
        '''
        face = face.cpu().numpy() #convert from tensor to numpy

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
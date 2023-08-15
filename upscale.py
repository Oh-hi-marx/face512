import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch
import os
from os import listdir
from os.path import isfile, join
import math 

def getFiles(mypath):
    onlyfiles = [join(mypath,f) for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles
    
class StableUpscaler:
    def __init__(self):
        # load model and scheduler
        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipeline.enable_attention_slicing()
        self.pipeline.set_use_memory_efficient_attention_xformers(True)
        self.pipeline = self.pipeline.to("cuda")
        
    def resizeSide(self, img, maxSize= 512):
        #resize by longest side. Keep ratio 
        w, h = img.size
        ratio =  max(w,h)/ maxSize
        if(ratio>1 ): #only resize if image larger than max size 
            newSize = [int(w/ratio), int(h/ratio)]
            print("Resizing to: ",newSize)
            img = img.resize(newSize)
        return img 
        
    def resizeRatio(self, img, maxSize ):
        maxSize *= 1000000
        #resize image keeping ratio the same. Maxsize is in megapixels
        w, h = img.size       
        while(w*h > maxSize):
            w -= 1
            h -= 1

        a,b = img.size
        if(a!= w):
            newSize = [int(w), int(h)]
            print("Resizing to: ",newSize)
            img = img.resize(newSize)
        return img
        
    def autoPrompt(self, img):
        prompt = "black | boot | brunette | bush | dress | hand | floor | girl | grass | pink | pose | short | sit | smile | sock | stocking | tight | wear | woman"
        prompt = "a young girl"

        prompt = "black tights"
        return prompt
    
    
    def upscaleImg(self, img, num_steps, maxSize , noise_level, prompt, fileName = None, guidance_scale = 3):
        h,w = img.size
        #resize to fit vram limitations
         #in megapixels. 
        img = self.resizeRatio(img, maxSize)

        #get prompt 
        #prompt = self.autoPrompt(img)
        print("Prompt: ", prompt)
        result = self.upscale(img, prompt , num_steps, noise_level,guidance_scale)
        
        if(fileName !=None):        
            promptCut = -1 if len(prompt) < 30 else 30
            img.save("inputs/results/" + fileName + "_step"+ str(num_steps) + "_size" + str(maxSize)+  "_prpt" + prompt[0:promptCut] +"_low.jpg")
        nh,nw = result .size
        if(nh!= 4*h or nw != 4*w): #ensure 4x upscale
            result  = result .resize((int(h*4), int(w*4)))
        return result , prompt
        
    def upscale(self, img, prompt, num_steps , noise_level = 20, guidance_scale= 3):
        #use guided stable 
        #input: PIL image
        #output: upscaled PIL image
              
        #https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale
        upscaled_image = self.pipeline(prompt=prompt, image=img, num_inference_steps = num_steps, noise_level = noise_level, guidance_scale  = guidance_scale).images[0]

        return upscaled_image
    
    def upscaleFolder(self, path, num_steps , maxSize, noise_level, prompt, guidance_scale):
        #create output folder
        outputPath = path + os.sep + "results"
        os.makedirs(outputPath, exist_ok = True)
        #get input files and loop through them
        files = getFiles(path)
        for file in files:     
            fileName =  file.split(os.sep)[-1].rsplit(".", 1)[0]
            print("Upscaling: ", file)
            img = Image.open(file).convert("RGB")
            result, prompt = self.upscaleImg(img, num_steps,maxSize, noise_level, prompt,guidance_scale = guidance_scale)
            promptCut = -1 if len(prompt) < 40 else 40
            result.save(outputPath + os.sep + fileName+ "_step"+ str(num_steps) + "_size" + str(maxSize)+  "_prpt" + prompt[0:promptCut] + ".jpg") 
            
if __name__ == "__main__":
    inputPath = "experiments/face"
    upscaler = StableUpscaler() 
    num_steps = 50
    maxSize = 0.2
    noise_level =25
    guidance_scale = 0
    prompt = "woman wearing black leather boots"
    prompt = 'a close of an young asian womans face'
    prompt = 'a young woman wearing earrings smiling and holding a cell phone'
    upscaler.upscaleFolder(inputPath, num_steps, maxSize , noise_level , prompt,guidance_scale)
import cv2
import numpy as np
import math 

def squareBox(box):
    boxWidth = box[2] - box[0]
    boxHeight = box[3] - box[1]
    #expand and create square aspect ratio based on longest side of bbox
    if(boxWidth> boxHeight):
        diff = int((boxWidth -boxHeight)/2)
        box = [box[0], box[1]-diff, box[2], box[3]+diff]
    elif(boxWidth< boxHeight):
        diff = int((boxHeight-boxWidth)/2)
        box = [box[0]-diff, box[1], box[2]+diff, box[3]]
    return box

def expandBox(box, expansion):
    boxWidth = box[2] - box[0]
    boxHeight = box[3] - box[1]
    box = [box[0]-int(boxWidth*expansion) , box[1]-int(boxHeight*expansion) , box[2]+int(boxWidth*expansion) , box[3]+int(boxHeight*expansion) ]
    return box 

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
    
def findEyeAngle(points):
    lefteye= points[0]
    righteye = points[1]
    nose = points[2]

    #precise angle alignment
    #do trig
    opposite = righteye[1] - lefteye[1]
    adjacent = righteye[0] - lefteye[0]
    tan = opposite / adjacent
    
    eyeAngle = math.atan(tan)   # angle = tan (o/h)
    eyeAngle = math.degrees(eyeAngle)  #radians -> degrees
    return eyeAngle

def addWithMask(img, background, mask):
    #adds a rgb, rgb, and mask image
    #add mask onto resized face as alpha layer
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img[:,:,3] = mask
    add_transparent_image(background,img)


def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    #adds a rgb and rgba image
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

def unrotateAndPaste(restoredFace, image, eyeAngle, box):
    #check restoredFace is square
    restoredFace = squareImg(restoredFace)
    mask = np.zeros(restoredFace.shape[0:2], np.uint8)
    mask.fill(255)
    
    derotated = rotate_image(restoredFace, -eyeAngle )
    mask = rotate_image(mask, -eyeAngle)

    #enlarge mask to hide aliases edges 
    h,w,_= restoredFace.shape
    mask = cv2.resize(mask, (int(h*0.99), int(w*0.99)))
    mh, mw = mask.shape
    white= np.zeros((h,w), np.uint8) 

    diff = h - mh
    white[int(diff/2):mh+int(diff/2),int(diff/2):mh+int(diff/2)] = mask
    mask = cv2.resize(white, (h, w))
     
    
    originalCrop = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    #originalCrop= cv2.resize(originalCrop, (h, w))
    originalCrop = cv2.resize(originalCrop, mask.shape[0:2])
    derotated = cv2.resize(derotated,(h, w))
    addWithMask(derotated , originalCrop, mask)

    
    return originalCrop

def squareImg(img): #does not keep aspect ratio
    h,w,c = img.shape 
    if(w!=h):
        new = int((w+h)/2)
        img= cv2.resize(img, (new, new))
    return img 

def resizeSide( img, maxSize= 512):
    #resize by longest side. Keep ratio 
    h,w = img.shape[0:2]
    ratio =  max(w,h)/ maxSize

    newSize = [int(w/ratio), int(h/ratio)]
    #print("Resizing to: ",newSize)
    img = cv2.resize(img, newSize)
    return img 
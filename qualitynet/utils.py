from io import BytesIO # "import StringIO" directly in python2
from PIL import Image
import random
from PIL import Image,ImageFilter,ImageEnhance
def jpegCompression(img, quality):
    # here, we create an empty string buffer    
    buffer = BytesIO()
    img.save(buffer, "JPEG", quality=quality)

    # ... do something else ...

    return Image.open(buffer)

def blur(img, min, max):
    if(min==max):
        blur = 0
    else:
        blur = random.randrange(min,max)
    
    img = img.filter(ImageFilter.GaussianBlur(radius = blur))
    return img

if __name__ == "__main__":
    for i in range(0,10,1):
        quality = i
        im1 = Image.open('faces/25.jpg')
        #img = jpegCompression(im1, quality)
        img = blur(im1, i, i+1)
        img.save(str(quality) + ".jpg")
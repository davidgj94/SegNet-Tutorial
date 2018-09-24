import sys
import numpy as np
from PIL import Image, ImageOps
import cv2
import pdb
print(cv2.__version__)

class VideoReader:
    
    def __init__(self, video_path, fps=1):
        self.vidcap = cv2.VideoCapture(video_path)
        self.count = 0
        self.delay_msec = int(1000 * (1 / fps))
        
    def read(self):
        
        if self.count > 0:
            self.vidcap.set(cv2.CAP_PROP_POS_MSEC,(self.count * self.delay_msec))    # added this line 
        success,image = self.vidcap.read()
        self.count = self.count + 1
        
        return success, image
        
def crop_imgs(img, crop_width=1024, crop_height=1080, border=(0,4)):
    
    img_width, img_height = img.shape[1], img.shape[0]
    img = Image.fromarray(img)

    img_batch = []
    for i_index, i in enumerate(np.arange(0,img_width, crop_width)):
        for j_index, j in enumerate(np.arange(0,img_height, crop_height)):
            img_aux = ImageOps.expand(img.crop((i,j,i + crop_width,j+crop_height)), border=border, fill='black')
            img_aux = np.array(img_aux)
            img_aux = np.transpose(img_aux, (2,0,1))
            img_aux = img_aux[(2,1,0),:,:]
            img_batch.append(img_aux)

    return np.array(img_batch)

def get_full_mask(masks, w=4, h=2, border=(0, 4)):

    num_masks = masks.shape[0]
    crop_width = masks.shape[2] - 2 * border[0]
    crop_height = masks.shape[1] - 2 * border[1]
    masks_ = masks[:,border[1]:border[1]+crop_height,border[0]:border[0]+crop_width]
    
    full_mask = np.zeros((crop_height * h, crop_width * w))
    n = 0
    for j in np.arange(w):
        
        for i in np.arange(h):
            
            full_mask[i*crop_height:(i+1)*crop_height, j*crop_width:(j+1)*crop_width] = masks_[n]
            n = n + 1
    
    return full_mask

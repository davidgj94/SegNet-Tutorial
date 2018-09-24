import sys
import numpy as np

import cv2
print(cv2.__version__)

class VideoReader:
    
    def __init__(self, video_path, fps):
        self.vidcap = cv2.VideoCapture(video_path)
        self.count = 0
        self.delay_msec = int(1000 * (1 / fps))
        
    def read(self):
        
        if self.count > 0:
            self.vidcap.set(cv2.CAP_PROP_POS_MSEC,(self.count * self.delay_msec))    # added this line 
        success,image = self.vidcap.read()
        self.count = self.count + 1
        
        return success, image
        
def crop_imgs(img, crop_width=1024, crop_height=1080):
    
    img = Image.fromarray(img)
    img_width, img_height = img.shape[1], img.shape[0]
    
    img_batch = []
    num_crops = 0
    for i_index, i in enumerate(np.arange(0,img_width, crop_width)):
        for j_index, j in enumerate(np.arange(0,img_height, crop_height)):
            img_aux = ImageOps.expand(img.crop((i,j,i + crop_width,j+crop_height)), border=(0,4), fill='black')
            img_batch.append(img_aux)
            num_crops += 1
    img_batch = np.array(img_batch)
    
    return np.reshape(img_batch, (num_crops,-1,-1,-1))

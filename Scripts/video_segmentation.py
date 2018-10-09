import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
from sklearn.preprocessing import normalize
import caffe; caffe.set_mode_gpu()
from canard.utils import crop_imgs, VideoReader, get_full_mask
import vis
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--video_path', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)

args = parser.parse_args()


net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

reader = VideoReader(args.video_path)
crop_height = 512
h = 2160 // crop_height
crop_weight = 540
w = 4096 // crop_weight
border = (0,2)
num_crops = h * w

count = 0
while true:
    
    success, img = reader.read()
    if success:
        input_batch = crop_imgs(img, crop_weight, crop_height, border)
        masks = np.zeros(num_crops, 1, crop_height, crop_weight)
        for i in range(num_crops):
            net.forward_all(data=input_batch[i])
            predicted = net.blobs['prob'].data
            output = np.squeeze(predicted[0,:,:,:])
            masks[i] = np.argmax(output, axis=0)
            
        full_mask = get_full_mask(masks, w, h, border)
        
        img = np.transpose(img, (1,2,0))
        img = img[:,:,(2,1,0)]
        
        vis_img = Image.fromarray(vis.vis_seg(img, full_mask, vis.make_palette(2))))
        vis_img.save(os.path.join(args.save_dir, 'frame_{}'.format(count)))
        count += count
    else:
        break

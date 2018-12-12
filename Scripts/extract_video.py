import numpy as np
import os.path
import argparse
from canard.utils import VideoReader
from canard.downscale import _downscale as downscale
from canard.upscale_mask import _upscale as upscale
import vis
import cv2
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--fps', type=float, default=1.0)
parser.add_argument('--max_dim', type=int, default=1000)

args = parser.parse_args()

reader = VideoReader(args.video_path, args.fps)

count = 0
while True:
    
    success, img = reader.read()

    if success:
    	print count
    	if args.max_dim == 4096:
    		cv2.imwrite(os.path.join(args.save_path, 'image_{}.png'.format(count)), img)
    	else:
	        img_down = downscale(img, args.max_dim, False, pad_img=False)
	        cv2.imwrite(os.path.join(args.save_path, 'image_{}.png'.format(count)), img_down)
        count += 1
    else:
        break

import _init_paths
import numpy as np
import os.path
import json
import scipy
import argparse
import math
import pylab
import vis
import sys
from PIL import Image
import caffe; caffe.set_mode_gpu()
import pdb
from canard.upscale_mask import _upscale as upscale
import cv2

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--imgs_txt', type=str, required=True)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--save_dir_mask', type=str)
parser.add_argument('--num_classes', type=str, default=3)
parser.add_argument('--max_dim', type=int, default=1000)
args = parser.parse_args()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

test_txt = np.loadtxt(args.imgs_txt, dtype=str)
test_names = [os.path.basename(img_path) for img_path in test_txt[:,1]]

for img_name in test_names:

	net.forward()

	image = net.blobs['data'].data
	label = net.blobs['label'].data
	predicted = net.blobs['prob'].data
	image = np.squeeze(image[0,:,:,:])
	output = np.squeeze(predicted[0,:,:,:])
	ind = np.argmax(output, axis=0)
	
	if args.save_dir:
		image = np.transpose(image, (1,2,0))
		image = image[:,:,(2,1,0)]
		vis_img = Image.fromarray(vis.vis_seg(image, ind, vis.make_palette(int(args.num_classes))))
		vis_img.save(os.path.join(args.save_dir, img_name))

	if args.save_dir_mask:
		cv2.imwrite(os.path.join(args.save_dir_mask, img_name), upscale(ind,args.max_dim))

import numpy as np
import os.path
import json
import scipy
import argparse
import math
import pylab
import vis
caffe_root = '/home/david/projects/caffe-segnet-cudnn5/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
from PIL import Image
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
parser.add_argument('--save_dir', type=str, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)


for i in range(0, args.iter):

	net.forward()

	image = net.blobs['data'].data
	label = net.blobs['label'].data
	predicted = net.blobs['prob'].data
	image = np.squeeze(image[0,:,:,:])
	output = np.squeeze(predicted[0,:,:,:])
	ind = np.argmax(output, axis=0)
	
	image = np.transpose(image, (1,2,0))
	image = image[:,:,(2,1,0)]
	
	vis_img = Image.fromarray(vis.vis_seg(image, ind, vis.make_palette(4)))
	vis_img.save(os.path.join(args.save_dir, 'image_{}.png'.format(i)))

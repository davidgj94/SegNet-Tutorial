#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 10:52:39 2018

@author: davidgj
"""
import _init_paths
import numpy as np
import os.path
import argparse
import vis
from PIL import Image
import caffe; caffe.set_mode_gpu()
from pathlib import Path
from canard.utils import crop_imgs, get_full_mask
import pdb
import parse

def get_full_mask_name(img_name):
    ext = os.path.splitext(img_name)[1]
    format_string = 'img_{:0>9}{:0>9}{:0>9}{:0>9}_{:0>9}:{:0>9}' + ext
    parsed = parse.parse(format_string, img_name)
    full_mask_name = 'img_{}{}{}{}'.format(parsed[0],parsed[1],parsed[2],parsed[3]) + ext
    return full_mask_name

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--imgs_txt', type=str, required=True)
parser.add_argument('--imgs_dir', type=str, required=True)
parser.add_argument('--num_classes', type=int, required=True)
parser.add_argument('--save_dir_blended', type=str, required=True)
parser.add_argument('--save_dir_mask', type=str, required=True)
args = parser.parse_args()

#p = Path(args.test_dir)
net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)
num_crops = 8
crop_width = 1024
crop_height = 1088

test_txt = np.loadtxt(args.imgs_txt, dtype=str)
test_names = [os.path.basename(img_path) for img_path in test_txt[:,1]]

k = 0
masks = np.zeros((num_crops, crop_height, crop_width))
for img_name in test_names:

    net.forward()

    predicted = net.blobs['prob'].data
    output = np.squeeze(predicted[0,:,:,:])
    masks[k] = np.argmax(output, axis=0)
    
    k += 1
    if k == 8:

        full_mask = get_full_mask(masks.astype("uint8"))
        full_mask_name = get_full_mask_name(img_name)

        Image.fromarray(full_mask).convert("RGB").save(os.path.join(args.save_dir_mask, full_mask_name))

        img = Image.open(os.path.join(args.imgs_dir, full_mask_name))
        vis_img = Image.fromarray(vis.vis_seg(np.asarray(img), full_mask.astype('uint8'), vis.make_palette(int(args.num_classes))))
        vis_img.save(os.path.join(args.save_dir_blended, full_mask_name))

        k = 0
        masks = np.zeros((num_crops, crop_height, crop_width))

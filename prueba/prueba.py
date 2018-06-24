import sys
import numpy as np
import os
from pathlib import Path
import shutil
from PIL import Image
import argparse
import vis
import caffe

tmp_dir = 'tmp/'
blended_dir = 'blended/'

if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
if os.path.exists(blended_dir):
    shutil.rmtree(blended_dir, ignore_errors=True)

os.makedirs(tmp_dir)
os.makedirs(blended_dir)
os.mknod(tmp_dir + 'test.txt')

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()

crop_height = 192

p = Path()
indices = [glob.parts[-1] for glob in p.glob('*.png')]

for img_name in indices:
    
    img = Image.open(img_name)
    
    width, height = img.size   # Get dimensions
    bottom = np.arange(crop_height, height, crop_height)
    top = bottom - crop_height
    
    num_crops = 0
    for index, (b, t) in enumerate(zip(bottom, top)):
        
        cropped_img = img.crop((0, int(t), 256, int(b)))
        new_name = '{}:{}.png'.format(os.path.splitext(img_name)[0], index)
            
        with open(tmp_dir + 'test.txt', 'a') as txt:
            txt.write(tmp_dir + new_name + ' ' + tmp_dir + new_name + '\n')
            
        cropped_img.save(tmp_dir + new_name)
        
        num_crops += 1
    
    net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)
    
    masks = []
    for i in range(0, num_crops):
        net.forward()
        predicted = net.blobs['prob'].data
        output = np.squeeze(predicted[0,:,:,:])
        ind = np.argmax(output, axis=0)
        masks.append(ind)
    
    #pdb.set_trace()
    total_mask = np.vstack(tuple(masks))
    img_cropped = np.array(img.crop((0, top[0], 256, bottom[-1])))
    vis_img = Image.fromarray(vis.vis_seg(img_cropped, total_mask, vis.make_palette(4)))
    vis_img.save(os.path.join(blended_dir + img_name))
    
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    os.mknod(tmp_dir + 'test.txt')
	
        
    
        



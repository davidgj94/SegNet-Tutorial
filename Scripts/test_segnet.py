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
from score_segnet import compute_hist, compute_confusion_matrix
import pdb
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--test_imgs', type=str, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

hist, acc, per_class_acc, per_class_iu = compute_hist(net, args.save_dir, args.test_imgs)


print '>>>','overall accuracy', acc

for idx, class_acc in enumerate(per_class_acc):
    print '>>>', 'Class {} accuracy:'.format(idx), class_acc
    
for idx, class_iu in enumerate(per_class_iu):
    print '>>>', 'Class {} iu:'.format(idx), class_iu
iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


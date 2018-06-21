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
from score_segnet import compute_hist, compute_confusion_matrix
import pdb
import caffe

def get_iter(glob):
    ext = os.path.splitext(glob.parts[-1])[1]
    format_string = 'iter_{:0>9}' + '.{}'.format(ext)
    parsed = parse.parse(format_string, glob.parts[-1])
    return int(parsed[0])

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--test_imgs', type=str, required=True)
parser.add_argument('--last_iter', type=int, default=0)
args = parser.parse_args()

results_path = os.path.join(args.save_dir,'results.p')
if os.path.exists(results_path):
    with open(results_path, 'rb') as f:
        hist, acc, per_class_acc, per_class_iu = pickle.load(f)
else:
    hist = []
    acc = []
    per_class_acc = []
    per_class_iu = []
    
    
caffe.set_mode_gpu()


p = Path(args.weights_dir)
states = sorted(list(p.glob('*.caffemodel')), key=get_iter)
states = states[args.last_iter:]

for state in states:
    
    net = caffe.Net(args.model, os.path.join(args.weights_dir, state.parts[-1]), caffe.TEST)

    hist_, acc_, per_class_acc_, per_class_iu_ = compute_hist(net, args.test_imgs)
    
    hist.append(hist_)
    acc.append(acc_)
    per_class_acc.append(per_class_acc_)
    per_class_iu.append(per_class_iu_)

    iter_name = os.path.splitext(state.parts[-1])[0]
    
    print '----', iter_name, '----'

    print '>>>','overall accuracy', acc_

    for idx, class_acc in enumerate(per_class_acc_):
        print '>>>', 'Class {} accuracy:'.format(idx), class_acc
        
    for idx, class_iu in enumerate(per_class_iu_):
        print '>>>', 'Class {} iu:'.format(idx), class_iu
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    
    with open(results_path, 'wb') as f:
        pickle.dump((hist, acc, per_class_acc, per_class_iu), f)

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
from score_segnet import compute_hist
import pdb
import caffe; caffe.set_mode_gpu()
from pathlib import Path
import parse
import pickle

def get_iter(glob):
    ext = os.path.splitext(glob.parts[-1])[1]
    format_string = 'snapshot_iter_{:0>9}' + '{}'.format(ext)
    parsed = parse.parse(format_string, glob.parts[-1])
    return int(parsed[0])

def select_iter(globs, iter_):
    states = []
    for glob in globs:
        if get_iter(glob) == iter_:
            states.append(glob)
            break
    return states

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights_dir', type=str, required=True)
parser.add_argument('--models_dir', type=str, required=True)
parser.add_argument('--iteration', type=int, default=None)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--test_imgs', type=str, required=True)
args = parser.parse_args()

p = Path(args.models_dir)

if args.iteration:
    results_path = os.path.join(args.save_dir,'results_{}.p'.format(args.iteration))
    hist = []
    acc = []
    per_class_acc = []
    per_class_iu = []
    states = select_iter(p.glob('*.caffemodel'), int(args.iteration))
else:
    results_path = os.path.join(args.save_dir,'results.p')
    
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            hist, acc, per_class_acc, per_class_iu = pickle.load(f)
    else:
        hist = []
        acc = []
        per_class_acc = []
        per_class_iu = []
        
    states = sorted(list(p.glob('*.caffemodel')), key=get_iter)
    last_iter = len(acc)
    states = states[last_iter:]
        
num_iter = len(list(Path(args.test_imgs).glob('*.png')))

for state in states:
    
    iter_name = os.path.splitext(state.parts[-1])[0]
    
    net = caffe.Net(args.model, os.path.join(args.weights_dir, iter_name, 'test_weights.caffemodel'), caffe.TEST)

    hist_, acc_, per_class_acc_, per_class_iu_ = compute_hist(net, num_iter)
   
    hist.append(hist_)
    acc.append(acc_)
    per_class_acc.append(per_class_acc_)
    per_class_iu.append(per_class_iu_)
    
    # Print results
    
    print '----', iter_name, '----'

    print '>>>','overall accuracy', acc_

    for idx, class_acc in enumerate(per_class_acc_):
        print '>>>', 'Class {} accuracy:'.format(idx), class_acc
        
    for idx, class_iu in enumerate(per_class_iu_):
        print '>>>', 'Class {} iu:'.format(idx), class_iu
    
    print '>>>','confusion matrix'
    print hist_ / hist_.sum(1)[:, np.newaxis]
    
    # Save results    
    
    with open(results_path, 'wb') as f:
        pickle.dump((hist, acc, per_class_acc, per_class_iu), f)


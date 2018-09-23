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
from segnet_utils import get_iter
from segnet_plots import plot_confusion_matrix

def select_iter(globs, iter_):
    model = None
    for glob in globs:
        if get_iter(glob) == iter_:
            model = glob
            break
    return os.path.splitext(model.parts[-1])[0]

def make_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_model', type=str, required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--inference_dir', type=str, required=True)
    parser.add_argument('--training_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--test_imgs', type=str, required=True)
    return parser

def print_test_results(acc, per_class_acc, per_class_iu):
    
    print '>>>','overall accuracy', acc

    for idx, class_acc in enumerate(per_class_acc):
        print '>>>', 'Class {} accuracy:'.format(idx), class_acc
        
    for idx, class_iu in enumerate(per_class_iu):
        print '>>>', 'Class {} iu:'.format(idx), class_iu
    

if __name__ == '__main__':
    
    parser = make_parser()
    args = parser.parse_args()
    
    results_path = os.path.join(args.save_dir,'results_{}.p'.format(args.iteration))
    
    iter_name = select_iter(Path(args.training_dir).glob('*.caffemodel'), args.iteration)
    net = caffe.Net(args.inference_model, os.path.join(args.inference_dir, iter_name, 'test_weights.caffemodel'), caffe.TEST)
    
    num_test_imgs = len(list(Path(args.test_imgs).glob('*')))
    hist, acc, per_class_acc, per_class_iu = compute_hist(net, num_test_imgs)
    # Save results    
    
    with open(results_path, 'wb') as f:
        pickle.dump((hist, acc, per_class_acc, per_class_iu), f)

    plot_confusion_matrix(hist, ['background', 'edge'], os.path.join(args.save_dir,'conf_matrix_{}.png'.format(args.iteration)))

    # Print results

    print_test_results(acc, per_class_acc, per_class_iu)


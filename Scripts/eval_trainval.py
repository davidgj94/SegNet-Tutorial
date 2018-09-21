import numpy as np
import os.path
import json
import scipy
import argparse
import math
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
from segnet_plots import plot_results

def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--inference_model', type=str, required=True)
	parser.add_argument('--inference_dir', type=str, required=True)
	parser.add_argument('--training_dir', type=str, required=True)
	parser.add_argument('--save_dir', type=str, required=True)
	parser.add_argument('--test_imgs', type=str, required=True)
	return parser

def get_iter_names(training_dir, last_iter):
	models = sorted(list(Path(training_dir).glob('*.caffemodel')), key=get_iter)
	models = models[last_iter:]
	iter_names = [os.path.splitext(model.parts[-1])[0] for model in models]
	return iter_names


if __name__ == '__main__':

	parser = make_parser()
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
	    
	num_test_imgs = len(list(Path(args.test_imgs).glob('*.png')))
	iter_names = get_iter_names(args.training_dir, len(acc))

	for iter_name in iter_names:
	    
	    net = caffe.Net(args.inference_model, os.path.join(args.inference_dir, iter_name, 'test_weights.caffemodel'), caffe.TEST)

	    hist_, acc_, per_class_acc_, per_class_iu_ = compute_hist(net, num_test_imgs)
	   
	    hist.append(hist_)
	    acc.append(acc_)
	    per_class_acc.append(per_class_acc_)
	    per_class_iu.append(per_class_iu_)
	    
	# Save results    

	with open(results_path, 'wb') as f:
	    pickle.dump((hist, acc, per_class_acc, per_class_iu), f)

	plot_results(acc, per_class_acc, per_class_iu, ['background', 'edges'] , args.save_dir)

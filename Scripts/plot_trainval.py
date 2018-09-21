import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import shutil
import argparse
import pdb
import numpy as np
from segnet_plots import plot_results_vs

parser = argparse.ArgumentParser()
parser.add_argument('--train_results', type=str)
parser.add_argument('--val_results', type=str)
parser.add_argument('--save_dir', type=str, required=True)
args = parser.parse_args()

results_path_train = os.path.join(args.train_results,'results.p')
results_path_val = os.path.join(args.val_results,'results.p')

with open(results_path_train, 'rb') as f:
	_, acc_train, per_class_acc_train, per_class_iu_train = pickle.load(f)
with open(results_path_val, 'rb') as f:
	_, acc_val, per_class_acc_val, per_class_iu_val = pickle.load(f)    
            
plot_results_vs(acc_train, acc_val, per_class_acc_train, per_class_acc_val, per_class_iu_train, per_class_iu_val, ['background', 'edges'], args.save_dir)


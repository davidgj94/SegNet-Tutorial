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
import sys
import pdb
from score_segnet import fast_hist

gt_mask_path = sys.argv[1]
predicted_mask_path = sys.argv[2]
cm = np.zeros((2,2))
p = Path(predicted_mask_path)
for glob in p.glob('*'):
	predicted_mask = np.asarray(Image.open(os.path.join(predicted_mask_path,glob.parts[-1])))
	gt_mask = np.asarray(Image.open(os.path.join(gt_mask_path,glob.parts[-1])))
	cm += fast_hist(gt_mask.flatten(), predicted_mask[:,:,0].flatten(), 2)


per_class_acc = np.diag(cm) / cm.sum(1)
print per_class_acc



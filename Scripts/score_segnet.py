from __future__ import division
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image
import vis
import shutil
from pathlib import Path
import pdb
import pickle

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, num_iter, layer='prob', gt='label'):
        
    n_cl = net.blobs[layer].channels
    hist = np.zeros((n_cl, n_cl))

    for i in range(num_iter):
        
        net.forward()
        pdb.set_trace()
        img_hist = fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)
        hist += img_hist
            
    acc = np.diag(hist).sum() / hist.sum()
    per_class_acc = np.diag(hist) / hist.sum(1)
    per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        
    return hist, acc, per_class_acc, per_class_iu

from __future__ import division
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image
import vis
from sklearn.metrics import confusion_matrix
import shutil
from pathlib import Path
import pdb
import pickle

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, imgs_path, save_dir=None, layer='prob', gt='label'):
    
    if save_dir:
        shutil.rmtree(save_dir, ignore_errors=True)
        os.mkdir(save_dir)
        blended_dir = os.path.join(save_dir, 'blended')
        os.mkdir(blended_dir)
        
    n_cl = net.blobs[layer].channels
    hist = np.zeros((n_cl, n_cl))
    
    p = Path(imgs_path)
    dataset = [glob.parts[-1] for glob in p.glob('*.png')]
    
    hist_list = []
    for idx in dataset:
        
        net.forward()
        img_hist = fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)
        hist += img_hist

        if save_dir:
            hist_list.append(img_hist)
            sat_path = os.path.join(imgs_path, idx)
            sat = Image.open(sat_path)
            plabel = net.blobs[layer].data[0].argmax(0)
            vis_img = Image.fromarray(vis.vis_seg(sat, plabel, vis.make_palette(4)))
            vis_img.save(os.path.join(blended_dir, idx))
    
    if save_dir:
        with open(os.path.join(save_dir, 'results.p'), 'wb') as f:
            pickle.dump(hist_list, f)
            
    acc = np.diag(hist).sum() / hist.sum()
    per_class_acc = np.diag(hist) / hist.sum(1)
    per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        
    return hist, acc, per_class_acc, per_class_iu


def compute_confusion_matrix(net, save_dir, imgs_path, layer='prob', gt='label'):
    
    n_cl = net.blobs[layer].channels
    if save_dir:
        shutil.rmtree(save_dir, ignore_errors=True)
    os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    #loss = 0
    p = Path(imgs_path)
    dataset = [glob.parts[-1] for glob in p.glob('*.png')]
    
    for idx in dataset:
        
        net.forward()
        label = net.blobs[gt].data
        predicted = net.blobs[layer].data
        output = np.squeeze(predicted[0,:,:,:])
        ind = np.argmax(output, axis=0)
        hist += confusion_matrix(label.flatten(), ind.flatten(), np.arange(n_cl))
        
        if save_dir:
            sat_path = os.path.join(imgs_path, idx)
            sat = Image.open(sat_path)
            vis_img = Image.fromarray(vis.vis_seg(sat, ind, vis.make_palette(4)))
            vis_img.save(os.path.join(save_dir, idx))
            
        # compute the loss as well
        #loss += net.blobs['loss'].data.flat[0]
    
    #loss = loss / len(dataset)
    acc = np.diag(hist).sum() / hist.sum()
    per_class_acc = np.diag(hist) / hist.sum(1)
    per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        
    return hist, acc, per_class_acc, per_class_iu

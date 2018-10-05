import caffe
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import shutil
import pickle
import time
from pathlib import Path
import parse
from argparse import ArgumentParser
import numpy
import scipy.io
from segnet_utils import get_iter, get_subdirs
from compute_bn_statistics import create_weights
import pdb
from PIL import Image

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass
    
def make_parser():
    p = ArgumentParser()
    p.add_argument('--solver', type=str)
    p.add_argument('--weights', type=str)
    return p

if __name__ == '__main__':

    # # init
    p = make_parser()
    args = p.parse_args()

    caffe.set_mode_gpu()

    solver = caffe.SGDSolver(args.solver)

    solver.net.copy_from(args.weights)

    solver.net.forward()
    img = solver.net.blobs['data'].data
    label = solver.net.blobs['label'].data
    #pdb.set_trace()

    img = np.squeeze(img)
    img = np.transpose(img, (1,2,0))
    img = img[:,:,(2,1,0)]
    img = Image.fromarray(img.astype('uint8'))
    img.save('img_prueba.png')

    print "End"
    

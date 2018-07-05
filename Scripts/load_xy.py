import caffe
import numpy as np
from PIL import Image
import random
import skimage.io
import matplotlib.pyplot as plt
from itertools import islice
from pathlib import Path
import random
import pdb

class XYLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - voc_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for PASCAL VOC semantic segmentation.

        example

        params = dict(voc_dir="/path/to/PASCAL/VOC2011",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="val")
        """
        # config
        params = eval(self.param_str)
        xy_dir = params['xy_dir']
        batchsize = int(params['batchsize'])
        if 'alfa' in params:
            alfa = float(params['alfa'])
        else:
            alfa = 1.0

        # two tops: data and label
        if len(top) != 1:
            raise Exception("Need to define one top: XY.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        
        X = Image.open('{}/{}.png'.format(xy_dir,'X'))
        X = np.array(X, dtype=np.float32)
        X = (X / 255) * alfa
        
        Y = Image.open('{}/{}.png'.format(xy_dir,'Y'))
        Y = np.array(Y, dtype=np.float32)
        Y = (Y / 192) * alfa
        
        XY = np.dstack((X,Y))
        XY = XY.transpose((2,0,1))
        
        self.batch = np.zeros((batchsize, 2, 192, 256))
        for idx in range(batchsize):
            self.batch[idx,:,:,:] = XY


    def reshape(self, bottom, top):
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(*self.batch.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.batch


    def backward(self, top, propagate_down, bottom):
        pass

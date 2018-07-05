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
        self.xy_dir = params['xy_dir']
        if 'alfa' in params:
            self.alfa = double(params['alfa'])
        else:
            self.alfa = 1.0

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: X and Y.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")


    def reshape(self, bottom, top):
        # load image + label image pair
        
        self.X = self.load_X()
        self.Y = self.load_Y()
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.X.shape)
        top[1].reshape(1, *self.Y.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.X
        top[1].data[...] = self.Y


    def backward(self, top, propagate_down, bottom):
        pass


    def load_X(self):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/{}.png'.format(self.xy_dir,'X'))
        im = np.array(im, dtype=np.float32)
        im = (im / 255) / self.alfa
        im = im[..., np.newaxis]
        im = im.transpose((2,0,1))
        return im


    def load_Y(self):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}/{}.png'.format(self.xy_dir,'X'))
        im = np.array(im, dtype=np.float32)
        im = (im / 191) / self.alfa
        im = im[..., np.newaxis]
        im = im.transpose((2,0,1))
        return im


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
from canard.utils import VideoReader

class VideoLayer(caffe.Layer):
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
        self.video_reader = VideoReader(params['video_path'], float(params['fps']))

        # two tops: data and label
        if len(top) != 1:
            raise Exception("Need to define one top.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")


    def reshape(self, bottom, top):
        # load image + label image pair
        
        self.data = self.load_frame()
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(*self.data.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data


    def backward(self, top, propagate_down, bottom):
        pass


    def load_frame(self):
        
        success, frame = self.video_reader.read()
        if success:
            frame = frame.transpose((2,0,1))
            frame = frame[np.newaxis,:]

        return frame

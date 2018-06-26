from PIL import Image
import numpy as np
import os
from pathlib import Path
import pdb
from argparse import ArgumentParser


def get_label_freqs(indices, num_classes):

    classPixelCount = np.zeros(num_classes)
    classTotalCount = np.zeros(num_classes)

    for idx in indices:
        
        label = np.array(Image.open(idx)).flatten()
        perImageFrequencies = np.bincount(label, minlength=num_classes)
        classPixelCount += perImageFrequencies
        nPixelsInImage = len(label)
        for i in np.arange(num_classes):
            if perImageFrequencies[i] > 0:
                classTotalCount[i] += nPixelsInImage
    return classPixelCount / classTotalCount

def get_weights(freqs):
    med_freq = np.median(freqs)
    return med_freq / freqs


def print_weights(weights):
    for weight in weights:
        print 'class_weighting:', weight

def make_parser():
    p = ArgumentParser()
    p.add_argument('--label_dir', type=str, required=True)
    p.add_argument('--num_classes', type=int, required=True)
    return p

p = make_parser()
args = p.parse_args()
path = Path(args.label_dir)
indices = [os.path.join(args.label_dir, glob.parts[-1]) for glob in path.glob('*.png')]
weights = get_weights(get_label_freqs(indices, args.num_classes))
print_weights(weights)
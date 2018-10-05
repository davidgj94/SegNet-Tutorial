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

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass
    
def make_parser():
    p = ArgumentParser()
    p.add_argument('--solver', type=str)
    p.add_argument('--train_model', type=str)
    p.add_argument('--weights', type=str)
    p.add_argument('--niter', type=int)
    p.add_argument('--nepoch', type=int)
    p.add_argument('--training_dir', type=str)
    p.add_argument('--inference_dir', type=str)
    return p

def get_states(training_dir):
    p = Path(training_dir)
    states = sorted(list(p.glob('*.solverstate')), key=get_iter)
    return states

def get_new_models(training_dir, inference_dir):
    p = Path(training_dir)
    models = sorted(list(p.glob('*.caffemodel')), key=get_iter)
    last_iter = len(get_subdirs(Path(inference_dir)))
    models = models[last_iter:]
    return models

if __name__ == '__main__':

    # # init
    p = make_parser()
    args = p.parse_args()

    caffe.set_mode_gpu()

    solver = caffe.SGDSolver(args.solver)

    states = get_states(args.training_dir)

    if states:
        solver.restore('/'.join(states[-1].parts))
    else:
        solver.net.copy_from(args.weights)

    for epoch in range(args.nepoch):
        solver.step(args.niter)

    del solver
    
    models = get_new_models(args.training_dir, args.inference_dir)

    for model in models:
        iter_name = os.path.splitext(model.parts[-1])[0]
        create_weights(args.train_model, os.path.join(args.training_dir, model.parts[-1]), os.path.join(args.inference_dir, iter_name))

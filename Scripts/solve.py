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

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass
    
def make_parser():
    p = ArgumentParser()
    p.add_argument('--solver', type=str)
    p.add_argument('--weights', type=str)
    p.add_argument('--niter', type=int, default=170)
    p.add_argument('--nepoch', type=int, default=5)
    p.add_argument('--snapshot_dir', type=str)
    return p

def get_iter(glob):
    ext = os.path.splitext(glob.parts[-1])[1]
    format_string = 'snapshot_iter_{:0>9}' + '{}'.format(ext)
    parsed = parse.parse(format_string, glob.parts[-1])
    return int(parsed[0])

# init
p = make_parser()
args = p.parse_args()

caffe.set_mode_gpu()

solver = caffe.SGDSolver(args.solver)

path = Path(args.snapshot_dir)
states = sorted(list(path.glob('*.solverstate')), key=get_iter)

if states:
    solver.restore('/'.join(states[-1].parts))
else:
    solver.net.copy_from(args.weights)
    mat = scipy.io.loadmat('../roads/ROADS/priors.mat')
    solver.net.params['scale-layer-0'][0].data[...] = mat['prior_0_norm']
    solver.net.params['scale-layer-1'][0].data[...] = mat['prior_1_norm']
    solver.net.params['scale-layer-2'][0].data[...] = mat['prior_2_norm']
    solver.net.params['scale-layer-3'][0].data[...] = mat['prior_3_norm']

for epoch in range(args.nepoch):
    solver.step(args.niter)

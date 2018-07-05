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

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass
    
def make_parser():
    p = ArgumentParser()
    p.add_argument('--solver')
    p.add_argument('--weights')
    return p

# init
p = make_parser()
args = p.parse_args()

caffe.set_mode_gpu()

solver = caffe.SGDSolver(args.solver)
solver.net.copy_from(args.weights)

niter = 170
nepoch = 5

for epoch in range(nepoch):
    solver.step(niter)

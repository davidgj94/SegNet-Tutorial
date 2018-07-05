import os
from argparse import ArgumentParser
import shutil

def make_parser():
    p = ArgumentParser()
    p.add_argument('--exp_name', type=str, required=True)
    return p

p = make_parser()
args = p.parse_args()

if os.path.exists('../Models/Training/{}'.format(args.exp_name)):
    shutil.rmtree('../Models/Training/{}'.format(args.exp_name), ignore_errors=True)
    
if os.path.exists('../Models/Inference/{}'.format(args.exp_name)):
    shutil.rmtree('../Models/Inference/{}'.format(args.exp_name), ignore_errors=True)
    
if os.path.exists('../results/{}'.format(args.exp_name)):
    shutil.rmtree('../results/{}'.format(args.exp_name), ignore_errors=True)

os.makedirs('../Models/Training/{}'.format(args.exp_name))
os.makedirs('../Models/Inference/{}'.format(args.exp_name))
os.makedirs('../results/{}'.format(args.exp_name))
os.makedirs('../results/{}/train'.format(args.exp_name))
os.makedirs('../results/{}/test'.format(args.exp_name))
os.makedirs('../results/{}/test/blended'.format(args.exp_name))
os.makedirs('../results/{}/val'.format(args.exp_name))

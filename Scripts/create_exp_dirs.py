import os
from argparse import ArgumentParser

def make_parser():
    p = ArgumentParser()
    p.add_argument('--exp_name', type=str, required=True)
    return p

p = make_parser()
args = p.parse_args()

os.makedirs('../Models/Training/{}'.format(args.exp_name))
os.makedirs('../Models/Inference/{}'.format(args.exp_name))
os.makedirs('../results/{}'.format(args.exp_name))
os.makedirs('../results/{}/train'.format(args.exp_name))
os.makedirs('../results/{}/test'.format(args.exp_name))
os.makedirs('../results/{}/test/blended'.format(args.exp_name))
os.makedirs('../results/{}/val'.format(args.exp_name))

import os
import pdb
from argparse import ArgumentParser

def create_dummt_txt_(base_txt, dummy_imgs_dir):

    X_txt_path = '../X.txt'.format(base_txt)
    if os.path.exists(X_txt_path):
        os.remove(X_txt_path)
    os.mknod(X_txt_path)

    Y_txt_path = '../Y.txt'.format(base_txt)
    if os.path.exists(Y_txt_path):
        os.remove(Y_txt_path)
    os.mknod(Y_txt_path)

    X_dir = os.path.join(dummy_imgs_dir, 'X.png')
    Y_dir = os.path.join(dummy_imgs_dir, 'Y.png')

    base_txt_path = '../roads/ROADS/{}.txt'.format(base_txt)
    num_lines = sum(1 for line in open(base_txt_path))

    for i in range(num_lines):
        with open(X_txt_path, 'a') as X_txt:
            X_txt.write('{} {}\n'.format(X_dir, 0))
        with open(Y_txt_path, 'a') as Y_txt:
            Y_txt.write('{} {}\n'.format(Y_dir, 0))


def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--dummy_dir', type=str, required=True)
    return parser

p = make_parser()
args = p.parse_args()
create_dummt_txt_('train', args.dummy_dir)
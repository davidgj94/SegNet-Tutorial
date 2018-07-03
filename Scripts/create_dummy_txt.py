import os
import pdb
from argparse import ArgumentParser

def create_dummt_txt_(base_txt, dummy_imgs_dir):

    dummy_txt_path = '../{}-dummy.txt'.format(base_txt)
    if os.path.exists(dummy_txt_path):
        os.remove(dummy_txt_path)
    os.mknod(dummy_txt_path)
    #pdb.set_trace()

    X_dir = os.path.join(dummy_imgs_dir, 'X.png')
    Y_dir = os.path.join(dummy_imgs_dir, 'Y.png')

    base_txt_path = '../roads/ROADS/{}.txt'.format(base_txt)
    num_lines = sum(1 for line in open(base_txt_path))
    #pdb.set_trace()

    for i in range(num_lines):
        with open(dummy_txt_path, 'a') as txt:
            txt.write('{} {}\n'.format(X_dir, Y_dir))


def make_parser():
    parser = ArgumentParser()
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-val', action='store_true')
    parser.add_argument('--dummy_dir', type=str, required=True)
    return parser

p = make_parser()
args = p.parse_args()
if args.train:
    create_dummt_txt_('train', args.dummy_dir)
if args.test:
    create_dummt_txt_('test', args.dummy_dir)
if args.val:
    create_dummt_txt_('val', args.dummy_dir)
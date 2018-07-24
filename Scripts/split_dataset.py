from PIL import Image
import sys
import numpy as np
import os
from pathlib import Path
import shutil
import random
import pdb
from argparse import ArgumentParser

def print_label_freqs(split, label_freqs):
    print split
    for idx, freq in enumerate(label_freqs):
        print '>>>', 'Class {} freq:'.format(idx), freq

def get_label_counts(label, num_classes):
    cls_count = np.zeros(num_classes)
    label_v = np.array(label).flatten()

    for i in range(num_classes):
        cls_count[i] = len(np.where(label_v==i)[0])

    return cls_count, np.sum(cls_count)
    
    
def crop_roads(crop_height, crop_step, indices, desc_txt, desc_image_dir, desc_label_dir, num_clasess):
    
    label_counts = np.zeros(num_clasess)
    num_pixels = 0
   
    for img_name in indices:
    
        image_dir = png_images_dir + img_name 
        label_dir = segmentation_class_raw_dir + img_name 
        img = Image.open(image_dir)
        label = Image.open(label_dir)
        
        label_counts_, num_pixels_ = get_label_counts(label, len(label_counts))
        #pdb.set_trace()
        label_counts += label_counts_
        num_pixels += num_pixels_
        
        width, height = img.size   # Get dimensions
        bottom = np.arange(crop_height, height, crop_step)
        top = bottom - crop_height
        
        for index, (b, t) in enumerate(zip(bottom, top)):
            
            cropped_img = img.crop((0, int(t), 256, int(b)))
            cropped_label = label.crop((0, int(t), 256, int(b)))
            new_name = '{}:{}'.format(os.path.splitext(img_name)[0], index)
                
            with open(dataset_dir + desc_txt, 'a') as txt:
                txt.write(desc_image_dir  + '{}.png'.format(new_name) + ' ' + desc_label_dir + '{}.png'.format(new_name) + '\n')
                
            cropped_img.save(desc_image_dir  + '{}.png'.format(new_name))
            cropped_label.save(desc_label_dir + '{}.png'.format(new_name))
    
    label_freqs = label_counts / num_pixels

    return label_freqs

def make_parser():
    p = ArgumentParser()
    p.add_argument('--labeled_roads', type=str, required=True)
    p.add_argument('--crop_height', type=int, default=192)
    p.add_argument('--crop_step', type=int, default=100)
    p.add_argument('--train_split', type=float, required=True)
    p.add_argument('--val_split', type=float, required=True)
    p.add_argument('--save_dir', type=str, required=True)
    p.add_argument('--num_classes', type=int)
    return p

def split_indices(args, indices):

    train_size = int(args.train_split * len(indices))
    val_size = int(args.val_split * len(indices))
    test_size = len(indices) - (train_size + val_size)

    train_indices = indices[:train_size]
    label_freqs_train = crop_roads(args.crop_height, args.crop_step, train_indices, 'train.txt', train_dir, trainannot_dir, args.num_classes)
    print_label_freqs('train', label_freqs_train)

    if(val_size > 0):
        val_indices = indices[train_size:(train_size + val_size)]
        label_freqs_val = crop_roads(args.crop_height, args.crop_step, val_indices, 'val.txt', val_dir, valannot_dir, args.num_classes)
        print_label_freqs('val', label_freqs_val)

    if(test_size > 0):
        test_indices = indices[(train_size + val_size):]
        label_freqs_test = crop_roads(args.crop_height, args.crop_step, test_indices, 'test.txt', test_dir, testannot_dir, args.num_classes)
        print_label_freqs('test', label_freqs_test)

#labeled_roads = sys.argv[1]
#crop_height = int(sys.argv[2])
#crop_step = int(sys.argv[3])

p = make_parser()
args = p.parse_args()

png_images_dir = args.labeled_roads + 'PNGImages/'
segmentation_class_raw_dir = args.labeled_roads + 'SegmentationClassRaw/'

_dataset_dir = '../{}/'.format(args.save_dir)
dataset_dir = _dataset_dir + 'ROADS/'
val_dir = dataset_dir + 'val/'
valannot_dir = dataset_dir + 'valannot/'
train_dir = dataset_dir + 'train/'
trainannot_dir = dataset_dir + 'trainannot/'
test_dir = dataset_dir + 'test/'
testannot_dir = dataset_dir + 'testannot/'

if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir, ignore_errors=True)

os.makedirs(dataset_dir)
os.makedirs(val_dir)
os.makedirs(valannot_dir)
os.makedirs(train_dir)
os.makedirs(trainannot_dir)
os.makedirs(test_dir)
os.makedirs(testannot_dir)
os.mknod(dataset_dir + 'train.txt')
os.mknod(dataset_dir + 'val.txt')
os.mknod(dataset_dir + 'test.txt')

p = Path(png_images_dir)
indices = [glob.parts[-1] for glob in p.glob('*.png')]
arrow_indices = np.loadtxt('arrow.txt', dtype='str').tolist()
pdb.set_trace()
indices = list(set(indices) - set(arrow_indices))
random.shuffle(indices)
random.shuffle(arrow_indices)
pdb.set_trace()


split_indices(args, indices)
split_indices(args, arrow_indices)


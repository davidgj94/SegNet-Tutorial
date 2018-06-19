from PIL import Image
import sys
import numpy as np
import os
from pathlib import Path
import shutil
import random

def print_label_freqs(split, label_freqs):
    print split
    for idx, freq in enumerate(label_freqs):
        print '>>>', 'Class {} freq:'.format(idx), freq

def get_label_freqs(label):
    cls_freq = []
    label_v = np.array(label).flatten()
    num_cls = np.unique(label_v).size

    for i in range(num_cls):
        cls_num.append(len(np.where(train_labels==i)[0]))

    cls_freq = np.array(cls_freq)
    cls_freq = cls_freq / np.sum(cls_freq)

    return v
    
    
def crop_roads(crop_height, crop_step, indices, desc_txt, desc_image_dir, desc_label_dir):
    
    label_freqs = []
    
    for img_name in indices:
    
    image_dir = png_images_dir + img_name 
    label_dir = segmentation_class_raw_dir + img_name 
    img = Image.open(image_dir)
    label = Image.open(label_dir)
    
    label_freqs.append(get_label_freqs(img,label))
    
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
        
    return label_freqs


labeled_roads = sys.argv[1]
png_images_dir = labeled_roads + 'PNGImages/'
segmentation_class_raw_dir = labeled_roads + 'SegmentationClassRaw/'

_dataset_dir = '../roads/'
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

crop_height = int(sys.argv[2])
crop_step = int(sys.argv[3])

p = Path(png_images_dir)
indices = [glob.parts[-1] for glob in p.glob('*.png')]
random.shuffle(indices)

train_split = 0.65
val_split = 0.1
train_size = int(train_split * len(indices))
val_size = int(val_split * len(indices))
test_size = len(indices) - (train_size + val_size)

train_indices = indices[:train_size]
label_freqs_train = crop_roads(crop_height, crop_step, train_indices, 'train.txt', train_dir, trainannot_dir)
print_label_freqs('train', label_freqs_train)

val_indices = indices[train_size:(train_size + val_size)]
label_freqs_val = crop_roads(crop_height, crop_step, val_indices, 'val.txt', val_dir, valannot_dir)
print_label_freqs('val', label_freqs_val)

if(test_size > 0):
    test_indices = indices[(train_size + val_size):]
    label_freqs_test = crop_roads(crop_height, crop_step, test_indices, 'test.txt', test_dir, testannot_dir)
    print_label_freqs('test', label_freqs_test)


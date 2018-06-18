from PIL import Image
import sys
import numpy as np
import os
from pathlib import Path
import shutil

labeled_roads = sys.argv[1]
png_images_dir = labeled_roads + 'PNGImages/'
segmentation_class_raw_dir = labeled_roads + 'SegmentationClassRaw/'

_dataset_dir = '/home/david/projects/SegNet-Tutorial/roads/'
dataset_dir = _dataset_dir + 'ROADS/'
val_dir = dataset_dir + 'val/'
valannot_dir = dataset_dir + 'valannot/'
train_dir = dataset_dir + 'train/'
trainannot_dir = dataset_dir + 'trainannot/'

if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir, ignore_errors=True)
os.makedirs(dataset_dir)
os.makedirs(val_dir)
os.makedirs(valannot_dir)
os.makedirs(train_dir)
os.makedirs(trainannot_dir)
os.mknod(dataset_dir + 'train.txt')
os.mknod(dataset_dir + 'val.txt')

crop_height = int(sys.argv[2])
crop_step = int(sys.argv[3])

p = Path(png_images_dir)
indices = [glob.parts[-1] for glob in p.glob('*.png')]

train_split = 0.7
train_size = 0
val_size = 0
idx = 0
num_roads = 0
roads_mean = np.array([0.0, 0.0, 0.0])

for img_name in indices:
    
    image_dir = png_images_dir + img_name 
    label_dir = segmentation_class_raw_dir + img_name 
    img = Image.open(image_dir)
    label = Image.open(label_dir)
    
    num_roads += 1
    roads_mean += np.mean(np.array(img), axis=(0, 1))
    
    width, height = img.size   # Get dimensions
    bottom = np.arange(crop_height, height, crop_step)
    top = bottom - crop_height
    
    idx += 1
    
    for index, (b, t) in enumerate(zip(bottom, top)):
        
        cropped_img = img.crop((0, int(t), 256, int(b)))
        cropped_label = label.crop((0, int(t), 256, int(b)))
        new_name = '{}:{}'.format(os.path.splitext(img_name)[0], index)

        if idx % 3 != 0:
            train_size += 1
            desc_txt = 'train.txt'
            desc_image_dir = train_dir
            desc_label_dir = trainannot_dir
        else:
            val_size += 1
            desc_txt = 'val.txt'
            desc_image_dir = val_dir
            desc_label_dir = valannot_dir
            
        with open(dataset_dir + desc_txt, 'a') as txt:
            txt.write(desc_image_dir  + '{}.png'.format(new_name) + ' ' + desc_label_dir + '{}.png'.format(new_name) + '\n')
            
        cropped_img.save(desc_image_dir  + '{}.png'.format(new_name))
        cropped_label.save(desc_label_dir + '{}.png'.format(new_name))
        

roads_mean /= num_roads
print('roads_mean: {}\n'.format(str(roads_mean))) #[109.31270171 112.73650684 107.62839719]
print('train_size: {}\n'.format(str(train_size)))
print('val_size: {}\n'.format(str(val_size)))
print('trainval_size: {}\n'.format(str(train_size + val_size)))


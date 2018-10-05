from PIL import Image, ImageOps
import sys
import numpy as np
import os
from pathlib import Path
import shutil
import random
import pdb
import os.path

img_dir = sys.argv[1]
label_dir = sys.argv[2]
txt_name = sys.argv[3]

img_dir_cropped = img_dir + '_cropped'
os.makedirs(img_dir_cropped)

label_dir_cropped = label_dir + '_cropped'
if img_dir != label_dir:
    os.makedirs(label_dir_cropped)

txt_path = os.path.join(str(Path(img_dir).parent), txt_name)
os.mknod(txt_path)

p = Path(img_dir);
for glob in p.glob('*'):
    
    img_name = glob.parts[-1]
    img = Image.open('{}/{}'.format(img_dir, img_name));
    label = Image.open('{}/{}'.format(label_dir, img_name));
    
    for i_index, i in enumerate(np.arange(0,4096,1024)):
        for j_index, j in enumerate(np.arange(0,2160,1080)):
            
            new_name = '{}_{}:{}.png'.format(os.path.splitext(img_name)[0], i_index, j_index)

            with open(txt_path, 'a') as txt:
                txt.write('{}/{}'.format(img_dir_cropped, new_name) + ' ' + '{}/{}'.format(label_dir_cropped, new_name) + '\n')
                
            img_aux = ImageOps.expand(img.crop((i,j,i + 1024,j+1080)), border=(0,4), fill='black')
            img_aux.save('{}/{}'.format(img_dir_cropped, new_name))

            if img_dir != label_dir:
                label_aux = ImageOps.expand(label.crop((i,j,i + 1024,j+1080)), border=(0, 4), fill='black')
                label_aux.save('{}/{}'.format(label_dir_cropped, new_name))
	


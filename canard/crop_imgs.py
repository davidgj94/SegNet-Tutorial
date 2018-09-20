from PIL import Image, ImageOps
import sys
import numpy as np
import os
from pathlib import Path
import shutil
import random
import pdb
from argparse import ArgumentParser

os.makedirs('train/')
os.makedirs('trainannot/')
os.mknod('train.txt')

p = Path('trainannot_old/');
for glob in p.glob('*.png'):

	img_name = glob.parts[-1]
	img = Image.open('train_old/{}'.format(img_name));
	label = Image.open('trainannot_old/{}'.format(img_name));

	for i_index, i in enumerate(np.arange(0,4096,1024)):
		for j_index, j in enumerate(np.arange(0,2160,1080)):

			img_aux = ImageOps.expand(img.crop((i,j,i + 1024,j+1080)), border=(0,4), fill='black')
			label_aux = ImageOps.expand(label.crop((i,j,i + 1024,j+1080)), border=(0, 4), fill='black')

			new_name = '{}_{}:{}.png'.format(os.path.splitext(img_name)[0], i_index, j_index)

			with open('train.txt', 'a') as txt:
				txt.write('../canard/train/{}'.format(new_name) + ' ' + '../canard/trainannot/{}'.format(new_name) + '\n')

			img_aux.save('train/' + new_name)
			label_aux.save('trainannot/' + new_name)
	


from pathlib import Path
import os
import sys

img_dir = sys.argv[1]
label_dir = sys.argv[2]
txt_name = sys.argv[3]

os.mknod(txt_name)

p = Path(img_dir)
for glob in p.glob('*'):

	img_name = glob.parts[-1]

	with open(txt_name, 'a') as txt:
		txt.write('../canard/{}/{}'.format(img_dir, img_name) + ' ' + '../canard/{}/{}'.format(label_dir, img_name) + '\n')

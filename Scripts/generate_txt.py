from pathlib import Path
import os
import sys
import pdb

img_dir = sys.argv[1]
label_dir = sys.argv[2]
txt_name = sys.argv[3]

txt_path = os.path.join(str(Path(label_dir).parent), txt_name)
os.mknod(txt_path)

p = Path(label_dir)
for glob in p.glob('*'):

    img_name = glob.parts[-1]
    
    with open(txt_path, 'a') as txt:
    	txt.write(os.path.join(img_dir, img_name) + ' ' + os.path.join(label_dir, img_name) + '\n')

import utils
import numpy as np
import pdb
from PIL import Image
from pathlib import Path

path = "/home/davidgj/projects_v2/SegNet-Tutorial/canard/train_old"
p = Path(path)
for glob in p.glob("*"):
	img = np.asarray(Image.open('{}/{}'.format(path, glob.parts[-1])))
	batch = utils.crop_imgs(img)
	masks_prueba = np.squeeze(batch[:,0,:,:])
	full_mask_prueba1 = img[:,:,-1]
	full_mask_prueba2 = utils.get_full_mask(masks_prueba)
	pdb.set_trace()

print "End"

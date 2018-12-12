import _init_paths
import numpy as np
import os.path
import argparse
import caffe; caffe.set_mode_gpu()
from canard.utils import VideoReader
from canard.downscale import _downscale as downscale
from canard.upscale_mask import _upscale as upscale
import vis
import cv2
import pdb
import time

def remove_noise(img, min_size=150):

    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--video_path', type=str, required=True)
parser.add_argument('--save_path_video', type=str, required=True)
parser.add_argument('--save_path_mask', type=str)
parser.add_argument('--save_path_blended', type=str)
parser.add_argument('--max_dim', type=int, default=1000)
parser.add_argument('--fps', type=float, default=1.0)
parser.add_argument('--num_classes', type=int, default=3)

args = parser.parse_args()

palette = np.array([[255,255,255],[0, 0, 255],[0, 255, 0]])


net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

H = 2160
W = 4096
reader = VideoReader(args.video_path, args.fps)
out = cv2.VideoWriter(args.save_path_video, cv2.VideoWriter_fourcc(*'XVID'), args.fps, (W,H))

count = 0
start = time.time()
while True:
    
    success, img = reader.read()

    if success:

        print count

        img_down = downscale(img, args.max_dim, False)
        img_down = img_down.transpose((2,0,1))
        img_down = np.asarray([img_down])
        net.blobs['data'].data[...] = img_down

        net.forward()

        predicted = net.blobs['prob'].data
        output = np.squeeze(predicted[0,:,:,:])
        mask = np.argmax(output, axis=0)
        new_mask = np.zeros(mask.shape)
        for ii in range(args.num_classes):
            mask_class = np.zeros(mask.shape)
            mask_class[mask == ii] = 255
            mask_class = np.uint8(mask_class)
            mask_class = remove_noise(mask_class, min_size=200)
            new_mask[mask_class == 255] = ii
        new_mask = np.uint8(new_mask)
        mask_up = upscale(new_mask, args.max_dim)

        if args.save_path_mask:
            cv2.imwrite(os.path.join(args.save_path_mask, 'image_{}.png'.format(count)), mask_up)

        if args.save_path_blended:
            vis_img_down = vis.vis_seg(img_down[...,::-1], mask, vis.make_palette(args.num_classes))
            cv2.imwrite(os.path.join(args.save_path_blended, 'image_{}.png'.format(count)), vis_img_down)

        #vis_img = vis.vis_seg(img[...,::-1], mask_up, vis.make_palette(args.num_classes))
        vis_img = vis.vis_seg(img[...,::-1], mask_up, palette)
        out.write(vis_img[...,::-1])

        count += 1

    else:
        out.release()
        end = time.time()
        break

elapsed_seg = end - start
elapsed_min = int(elapsed_seg / 60)
remaining_seg = int(elapsed_seg - elapsed_min * 60)
print '%30s' % 'Executed SegNet in ', str(elapsed_min), 'min', str(remaining_seg), 'seg'


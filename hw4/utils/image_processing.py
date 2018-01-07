import numpy as np
from scipy import misc
import random
import skimage
import skimage.io
import skimage.transform

def load_image_array(image_file, image_size):
    img = skimage.io.imread(image_file)
    if len(img.shape) == 2:
        img_new = np.ndarray((img.shape[0], img.shape[1], 3), dtype='uint8')
        img_new[:,:,0] = img
        img_new[:,:,1] = img
        img_new[:,:,2] = img
        img = img_new

    img_resized = skimage.transform.resize(img, (image_size, image_size), mode='constant')

    set_flip = True

    if set_flip and random.random() > 0.5:
        img_resized = np.fliplr(img_resized)

    return img_resized.astype('float32')

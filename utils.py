from PIL import Image
import numpy as np

BGR_MEANS = np.array([103.939, 116.779, 123.68], dtype = np.float64)

# Read and swap RGB to BGR as input

def load_image(path):
    im = Image.open(path)
    im_data = np.array(im)
    # if black and white images, add condition.
    red = np.copy(im_data[:, :, :0])
    blue = np.copy(im_data[:, :, :2])
    im_data[:, :, 0] = blue
    im_data[:, :, 2] = red
    centered_im = im_data - BGR_MEANS
    print (centered_im.shape)
    return centered_im

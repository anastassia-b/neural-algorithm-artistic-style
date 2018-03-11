from PIL import Image
import numpy as np

#debugging!
import pdb

# Constant from the Gatys paper
BGR_MEANS = np.array([103.939, 116.779, 123.68], dtype = np.float64)

def load_image(path):
    image = Image.open(path)
    image_data = np.array(image)
    # if black and white images, add condition.

    # Read and swap RGB to BGR as input
    red = np.copy(image_data[:, :, 0])
    blue = np.copy(image_data[:, :, 2])

    image_data[:, :, 0] = blue
    image_data[:, :, 2] = red

    centered_image = image_data - BGR_MEANS
    return centered_image

def save_image(path, image_data):
    image_data = image_data + BGR_MEANS
    blue = np.copy(image_data[:, :, 0])
    red = np.copy(image_data[:, :, 2])
    image_data[:, :, 0] = red
    image_data[:, :, 2] = blue
    # correcting the range
    image_data = np.clip(im_data, 0, 255)
    image_data = image_data.astype(np.uint8)
    image = Image.fromarray(image_data)
    image.save(path)

# image_data = load_image('./input/content_tubingen.jpg')
# save_image('./output/output_image.jpg', image_data)

# ValueError: could not broadcast input array from shape (768,1024,2) into shape (768,1024)

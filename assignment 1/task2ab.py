import matplotlib.pyplot as plt
import pathlib
import numpy as np
from utils import read_im, save_im
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "lake.jpg"))
plt.imshow(im)


def greyscale(im):
    """ Converts an RGB image to greyscale

    Args:
        im ([type]): [np.array of shape [H, W, 3]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    # First we will collect the height and width of the image
    height = len(im[:, 0, 0])
    width = len(im[0, :, 0])
    # Create an empty 2D numpy array with height and width of original image
    im_grey = np.empty([height, width])
    # Iterate over the empty 2D array just created and provide with weighted RGB values for same pixel position
    for h in range(height):
        for w in range(width):
            im_grey[h, w] = 0.212 * im[h, w, 0] + 0.7152 * im[h, w, 1] + 0.0722 * im[h, w, 2]
    # Return the converted grey scale image
    return im_grey


im_greyscale = greyscale(im)
save_im(output_dir.joinpath("lake_greyscale.jpg"), im_greyscale, cmap="gray")
plt.imshow(im_greyscale, cmap="gray")


def inverse(im):
    """ Finds the inverse of the greyscale image

    Args:
        im ([type]): [np.array of shape [H, W]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    # YOUR CODE HERE
    # First we will collect the height and width of the image
    height = len(im[:, 0])
    width = len(im[0, :])
    # Create an empty 2D numpy array with height and width of original image
    im_inverse = np.empty([height, width])
    # Iterate over the empty 2D array just created and provide with inverse value (1 - pixel_value) for same pixel position
    for h in range(height):
        for w in range(width):
            im_inverse[h,w] = np.max(im) - im[h,w] #if you want the range from the image
            #im_inverse[h,w] = 1.0 - im[h,w] #if you want the maximum possible range value
    # Returning the inverse of input greyscale image
    return im_inverse

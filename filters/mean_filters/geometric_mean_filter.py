"""
Implementation of arithmetic mean filter algorithm
"""
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey
import numpy as np
import time

def geometric_mean_filter(
        image: np.ndarray,
        mask: int
        ) -> np.ndarray:

    # set image borders
    bd = int(mask / 2)

    # copy image size
    geometric_mean_img = np.zeros_like(image)

    # loop over pixels in image
    for i in range(bd, image.shape[0] - bd):
        for j in range(bd, image.shape[1] - bd):

            # get kernal given size
            kernel = np.ravel(image[i - bd : i + bd + 1, j - bd : j + bd + 1])
            # calculate window gemetric mean
            geometric_mean = np.power(np.prod(kernel), 1/len(kernel))
            geometric_mean_img[i, j] = geometric_mean

    return geometric_mean_img


if __name__ == "__main__":

    # read original image
    img = imread("../image_data/lena.jpg")

    # turn image in gray scale value
    gray = cvtColor(img, COLOR_BGR2GRAY)

    # get values with two different mask size
    geometric_mean3x3 = geometric_mean_filter(gray, 3)
    geometric_mean5x5 = geometric_mean_filter(gray, 5)

    # show result images
    imshow("geometric mean filter with 3x3 mask", geometric_mean3x3)
    imshow("geometric mean filter with 5x5 mask", geometric_mean5x5)
    waitKey(0)

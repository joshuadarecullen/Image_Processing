"""
Implementation of harmonic mean filter algorithm
"""
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey
import numpy as np


def contraharmonic_mean_filter(
        image: np.ndarray,
        mask: int,
        Q: float) -> np.ndarray:

    # set image borders
    bd = int(mask / 2)

    # copy image size
    contraharmonic_mean_img = np.zeros_like(image)

    # loop over pixels in image
    for i in range(bd, image.shape[0] - bd):
        for j in range(bd, image.shape[1] - bd):

            # get kernal given size
            kernel = np.ravel(image[i - bd : i + bd + 1, j - bd : j + bd + 1])

            # calculate contraharmonic window mean sum(g(x,y)^Q+1) / sum(g(x,y)^Q)
            # Q is called the order of the filter.
            numerator = np.sum([np.power(v, Q+1) for v in kernel])
            denominator = np.sum([np.power(v, Q) for v in kernel])
            contrahamonic_mean = numerator / denominator
            contraharmonic_mean_img[i, j] = contrahamonic_mean

    return contraharmonic_mean_img


if __name__ == "__main__":

    # read original image
    img = imread("../image_data/lena.jpg")

    # turn image in gray scale value
    gray = cvtColor(img, COLOR_BGR2GRAY)

    # get values with two different mask size
    contraharmonic_mean3x3 = contraharmonic_mean_filter(gray, 3, 1)
    contraharmonic_mean5x5 = contraharmonic_mean_filter(gray, 5, 1)

    # show result images
    imshow("mean filter with 3x3 mask", contraharmonic_mean3x3)
    imshow("mean filter with 5x5 mask", contraharmonic_mean5x5)
    waitKey(0)

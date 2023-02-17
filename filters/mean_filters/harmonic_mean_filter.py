"""
Implementation of harmonic mean filter algorithm
"""
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey
import numpy as np


def harmonic_mean_filter(
        image: np.ndarray,
        mask: int
        ) -> np.ndarray:

    # set image borders
    bd = int(mask / 2)

    # copy image size
    harmonic_mean_img = np.zeros_like(image)

    # loop over pixels in image
    for i in range(bd, image.shape[0] - bd):
        for j in range(bd, image.shape[1] - bd):

            # get kernal given size
            kernel = np.ravel(image[i - bd : i + bd + 1, j - bd : j + bd + 1])

            # calculate harmonic window mean (mn / sum(1/g(r,c)) r,c the row and column of kernal
            harmonic_mean = len(kernel) / np.sum([1/v for v in kernel])
            harmonic_mean_img[i, j] = harmonic_mean

    return harmonic_mean_img


if __name__ == "__main__":

    # read original image
    img = imread("../image_data/lena.jpg")

    # turn image in gray scale value
    gray = cvtColor(img, COLOR_BGR2GRAY)

    # get values with two different mask size
    harmonic_mean3x3 = harmonic_mean_filter(gray, 3)
    harmonic_mean5x5 = harmonic_mean_filter(gray, 5)

    # show result images
    imshow("mean filter with 3x3 mask", harmonic_mean3x3)
    imshow("mean filter with 5x5 mask", harmonic_mean5x5)
    waitKey(0)

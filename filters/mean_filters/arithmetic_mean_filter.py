"""
Implementation of arithmetic mean filter algorithm
"""
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey
import numpy as np


def mean_filter(image: np.ndarray,
        mask: int) -> np.ndarray:

    # set image borders
    bd = int(mask / 2)

    # copy image size
    mean_img = np.zeros_like(image)

    # loop over pixels in image
    for i in range(bd, image.shape[0] - bd):
        for j in range(bd, image.shape[1] - bd):

            # get kernal given size
            kernel = np.ravel(image[i - bd : i + bd + 1, j - bd : j + bd + 1])

            # calculate window mean
            mean = np.mean(kernel)
            mean_img[i, j] = mean

    return mean_img


if __name__ == "__main__":

    # read original image
    img = imread("/home/joshua/Documents/university/2/spring-term/IP/Image_Processing/image_data/lena.jpg")

    # turn image in gray scale value
    gray = cvtColor(img, COLOR_BGR2GRAY)

    # get values with two different mask size
    mean3x3 = mean_filter(gray, 3)
    mean5x5 = mean_filter(gray, 5)

    # show result images
    imshow("mean filter with 3x3 mask", mean3x3)
    imshow("mean filter with 5x5 mask", mean5x5)
    waitKey(0)

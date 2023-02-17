"""
Implementation of alpha-trimmed mean filter algorithm

Suppose that we delete the d 2 lowest and the d 2 highest intensity values of g(r, c)
in the neighborhood Sxy . Let gR (r, c) represent the remaining mn − d pixels in Sxy .
A filter formed by averaging these remaining pixels is called an alpha-trimmed mean
filter.

form: ( 1 / mn - d ) * sum(g(r,c)) r,c row and column of kernel Sxy is the region of the image, same size as kernal

The value of d can range from 0 to mn − 1. When d = 0 the alpha-trimmed fil-
ter reduces to the arithmetic mean filter discussed earlier.
If we choose d = mn − 1, the filter becomes a median filter. For other values of d, the alpha-trimmed filter is
useful in situations involving multiple types of noise, such as a combination of salt-
and-pepper and Gaussian noise.
"""
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey

import numpy as np


def alpha_trim_mean_filter(gray_img: np.ndarray, mask: int, d: float) -> np.ndarray:
    """
    :param gray_img: gray image
    :param mask: mask size
    :return: image with midpoint filter
    """
    # set image borders
    bd = int(mask / 2)
    # copy image size
    atm_img = np.zeros_like(gray_img)
    for i in range(bd, gray_img.shape[0] - bd):
        for j in range(bd, gray_img.shape[1] - bd):

            # get mask according with mask
            kernel = np.ravel(gray_img[i - bd : i + bd + 1, j - bd : j + bd + 1])

            # calculate mask median
            atm = ( 1 / len(kernel) - d )  * np.sum(kernel)
            atm_img[i, j] = atm

    return atm_img


if __name__ == "__main__":
    # read original image
    img = imread("/home/joshua/Documents/university/2/spring-term/IP/Image_Processing/image_data/lena.jpg")
    # turn image in gray scale value
    gray = cvtColor(img, COLOR_BGR2GRAY)

    # get values with two different mask size
    atm3x3 = alpha_trim_mean_filter(gray, 3, 4)

    # show result images
    imshow("midpoint filter with 3x3 mask", atm3x3)
    waitKey(0)

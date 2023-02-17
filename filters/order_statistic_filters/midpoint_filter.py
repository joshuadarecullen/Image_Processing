"""
Implementation of midpoint filter algorithm
"""
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey

import numpy as np


def midpoint_filter(gray_img: np.ndarray, mask: int) -> np.ndarray:
    """
    :param gray_img: gray image
    :param mask: mask size
    :return: image with midpoint filter
    """
    # set image borders
    bd = int(mask / 2)
    # copy image size
    midpoint_img = np.zeros_like(gray_img)
    for i in range(bd, gray_img.shape[0] - bd):
        for j in range(bd, gray_img.shape[1] - bd):

            # get mask according with mask
            kernel = np.ravel(gray_img[i - bd : i + bd + 1, j - bd : j + bd + 1])

            # calculate mask median
            midpoint = (np.min(kernel) + np.max(kernel)) / 2
            midpoint_img[i, j] = midpoint

    return midpoint_img


if __name__ == "__main__":
    # read original image
    img = imread("/home/joshua/Documents/university/2/spring-term/IP/Image_Processing/image_data/lena.jpg")
    # turn image in gray scale value
    gray = cvtColor(img, COLOR_BGR2GRAY)

    # get values with two different mask size
    midpoint3x3 = midpoint_filter(gray, 3)
    midpoint5x5 = midpoint_filter(gray, 5)

    # show result images
    imshow("midpoint filter with 3x3 mask", midpoint3x3)
    imshow("midpoint filter with 5x5 mask", midpoint5x5)
    waitKey(0)

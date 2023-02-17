"""
Implementation of min and max filter algorithms
"""
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey
import numpy as np


def min_filter(gray_img: np.ndarray, mask: int) -> np.ndarray:
    """
    :param gray_img: gray image
    :param mask: mask size
    :return: image with min filter
    """
    # set image borders
    bd = int(mask / 2)
    # copy image size
    min_img = np.zeros_like(gray_img)

    for i in range(bd, gray_img.shape[0] - bd):
        for j in range(bd, gray_img.shape[1] - bd):

            # get mask according with mask
            kernel = np.ravel(gray_img[i - bd : i + bd + 1, j - bd : j + bd + 1])

            # calculate mask median
            min = np.min(kernel)
            min_img[i, j] = min

    return min_img


def max_filter(gray_img: np.ndarray, mask: int) -> np.ndarray:
    """
    :param gray_img: gray image
    :param mask: mask size
    :return: image with max filter
    """
    # set image borders
    bd = int(mask / 2)
    # copy image size
    max_img = np.zeros_like(gray_img)

    for i in range(bd, gray_img.shape[0] - bd):
        for j in range(bd, gray_img.shape[1] - bd):

            # get mask according with mask
            kernel = np.ravel(gray_img[i - bd : i + bd + 1, j - bd : j + bd + 1])

            # calculate mask median
            max = np.max(kernel)
            max_img[i, j] = max

    return max_img


if __name__ == "__main__":
    # read original image
    img = imread("/home/joshua/Documents/university/2/spring-term/IP/Image_Processing/image_data/lena.jpg")
    # turn image in gray scale value
    gray = cvtColor(img, COLOR_BGR2GRAY)

    # get values with two different mask size
    minpoint3x3 = min_filter(gray, 3)
    maxpoint3x3 = max_filter(gray, 3)

    # show result images
    imshow("Min filter with 3x3 mask", minpoint3x3)
    imshow("Max filter with 3x3 mask", maxpoint3x3)
    waitKey(0)

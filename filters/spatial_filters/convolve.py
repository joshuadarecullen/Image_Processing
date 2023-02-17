# @Author  : Joshua
# @File    : convolve.py
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey
import numpy as np

def im2col(image, block_size):

    rows, cols = image.shape
    dst_height = cols - block_size[1] + 1
    dst_width = rows - block_size[0] + 1
    image_array = np.zeros((dst_height * dst_width, block_size[1] * block_size[0]))

    row = 0
    for i in range(0, dst_height):
        for j in range(0, dst_width):
            window = np.ravel(image[i : i + block_size[0], j : j + block_size[1]])
            image_array[row, :] = window
            row += 1

    return image_array


def img_convolve(image, filter_kernel):

    height, width = image.shape[0], image.shape[1]
    k_size = filter_kernel.shape[0]
    pad_size = k_size // 2

    # Pads image with the edge values of array.
    image_tmp = np.pad(image, pad_size, mode="edge")

    # im2col, turn the k_size*k_size pixels into a row and np.vstack all rows
    image_array = im2col(image_tmp, (k_size, k_size))

    #  turn the kernel into shape(k*k, 1)
    kernel_array = np.ravel(filter_kernel)

    # reshape and get the dst image
    dst = np.dot(image_array, kernel_array).reshape(height, width)

    return dst


if __name__ == "__main__":

    # read original image
    img = imread(r"../image_data/lena.jpg")

    # turn image in gray scale value
    gray = cvtColor(img, COLOR_BGR2GRAY)

    # Laplace operator
    '''
    Because the Laplacian is a derivative operator, it highlights sharp intensity tran-
    sitions in an image and de-emphasizes regions of slowly varying intensities. This
    will tend to produce images that have grayish edge lines and other discontinuities,
    all superimposed on a dark, featureless background. Background features can be
    “recovered” while still preserving the sharpening effect of the Laplacian by adding
    the Laplacian image to the original
    '''
    Laplace_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    out = img_convolve(gray, Laplace_kernel).astype(np.uint8)

    imshow("Original", gray)
    imshow("Laplacian", out)
    waitKey(0)

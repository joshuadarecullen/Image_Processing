"""
Implementation of gaussian filter algorithm
"""
from itertools import product

# processing images module
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey, destroyAllWindows
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros, random
import time

# Is the intitial 
def gen_gaussian_kernel(k_size, sigma) -> np.ndarray:
    center = k_size // 2
    # grab matrix and assign to x and y
    x, y = mgrid[0 - center : k_size - center, 0 - center : k_size - center]
    g = 1 / (2 * pi * sigma) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
    return g


def gaussian_filter(image, k_size, sigma) -> np.ndarray:

    height, width = image.shape[0], image.shape[1]

    # dst image height and width
    dst_height = height - k_size + 1
    dst_width = width - k_size + 1

    # im2col, turn the k_size*k_size pixels into a row and np.vstack all rows
    image_array = zeros((dst_height * dst_width, k_size * k_size))
    row = 0
    # get the indexes for every pixel in the image
    for i, j in product(range(dst_height), range(dst_width)):
        window = ravel(image[i : i + k_size, j : j + k_size]) #  get kernel size matrix from image and reduce to vector
        image_array[row, :] = window
        row += 1

    # turn the kernel into shape(k*k, 1)
    gaussian_kernel = gen_gaussian_kernel(k_size, sigma)
    filter_array = ravel(gaussian_kernel)

    # reshape and get the dst image
    dst = dot(image_array, filter_array).reshape(dst_height, dst_width).astype(uint8)

    return dst


if __name__ == "__main__":
    # read original image
    img = imread(r"../image_data/lena.jpg")
    # turn image in gray scale value
    gray = cvtColor(img, COLOR_BGR2GRAY)

    # get values with two different mask size
    # gaussian3x3 = gaussian_filter(gray, 3, sigma=1)
    # gaussian5x5 = gaussian_filter(gray, 5, sigma=4) # blur

    noisy = gray + 0.8 * gray.std() * random.random(gray.shape)
    filtered5x5 = gaussian_filter(noisy, 5, sigma=0.8)

    # # show result images
    # imshow("gaussian filter with 3x3 mask", gaussian3x3)
    imshow("Original Image", gray)
    imshow("Noisy Image", noisy.astype(uint8))
    imshow("Gaussian Filter with 5x5 Mask", filtered5x5)
    waitKey(0)
    destroyAllWindows()

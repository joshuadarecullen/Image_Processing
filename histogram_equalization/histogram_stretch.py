"""
Created on 2022

@author: Joshua
"""
import copy
import os

import cv2
import numpy as np
import time
from matplotlib import pyplot as plt


class contrastStretch:
    def __init__(self) -> None:
        self.img = ""
        self.original_image = ""
        self.last_list = []
        self.rem = 0
        self.L = 256
        self.sk = 0
        self.k = 0
        self.number_of_rows = 0
        self.number_of_cols = 0

    def stretch(self, input_image) -> cv2.imread:

        self.img = cv2.imread(input_image, 0) #  using 0 to read image in grey scale mode
        self.original_image = copy.deepcopy(self.img)

        x, _, _ = plt.hist(self.img.ravel(), 256, [0, 256], label="x") #  get frequency of pixels for each colour in the rgb colour range (256)
        self.k = np.sum(x) # total amount of pixels

        # create the new greyscale colour assosciated with each num 1-256
        for i in range(len(x)):

            prk = x[i] / self.k #  normalises between 0 and 1 and is the probability of pixels at that value
            self.sk += prk # accumulate: will eventually equal one
            last = (self.L - 1) * self.sk #  256-1 * cummulative probability up to the pixel value

            # for use in first iteration
            if self.rem != 0:
                self.rem = int(last % last)

            last = int(last + 1 if self.rem >= 0.5 else last) # round to nearest int
            self.last_list.append(last) # append new colour for given rgb

        self.number_of_rows = int(np.ma.count(self.img) / self.img[1].size) # get number of rows in image
        self.number_of_cols = self.img[1].size

        # set each pixel to new value
        for i in range(self.number_of_cols):
            for j in range(self.number_of_rows):
                num = self.img[j][i]
                if num != self.last_list[num]:
                    self.img[j][i] = self.last_list[num]

        # cv2.imwrite("output_data/output.jpg", self.img)

        return self.img

    def plotHistogram(self):
        plt.hist(self.img.ravel(), 256, [0, 256])
        plt.title('Post Stretch Contrast')
        plt.xlabel('Pixel Grey-Scale Values (1-256)')
        plt.ylabel('Frequency of Pixels')
        plt.show()

    def showImage(self):
        print('Press 0 to EXIT IMAGES NOT THE EXIT BUTTON ON GUI!')
        cv2.imshow("Output-Image", self.img)
        cv2.imshow("Input-Image", self.original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    file_path = f"./image_data/input.jpg"
    stretcher = contrastStretch()
    stretcher.stretch(file_path)
    stretcher.plotHistogram()
    stretcher.showImage()

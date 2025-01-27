import numpy as np
import cv2
from matplotlib import pyplot as plt


class Sobel():

    def __init__(self):
        pass

    def sobelX(self, img):
        container = np.copy(img)
        size = container.shape
        for i in range(1, size[0] - 1):
            for j in range(1, size[1] - 1):
                gx = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - \
                    (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
                container[i][j] = min(255, np.abs(gx))
        return container

    def sobelY(self, img):
        container = np.copy(img)
        size = container.shape
        for i in range(1, size[0] - 1):
            for j in range(1, size[1] - 1):
                gy = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - \
                    (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
                container[i][j] = min(255, np.sqrt(gy**2))
        return container

    def Magnitude(self, img1, img2):
        container = np.copy(img1)
        size = container.shape
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                container[i][j] = min(255, np.sqrt(
                    img1[i][j]**2 + img2[i][j]**2))
        return container

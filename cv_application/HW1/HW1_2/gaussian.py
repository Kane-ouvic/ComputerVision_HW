import math
import numpy as np


class GaussianBlur():

    def __init__(self, radius=1, sigma=1.5):
        self.radius = radius
        self.sigma = sigma

    # 高斯的計算公式
    def calc(self, x, y):
        res1 = 1 / (2*math.pi*self.sigma*self.sigma)
        res2 = math.exp(-(x*x + y*y)/(2*self.sigma*self.sigma))
        return res1 * res2

    # 產生高斯矩陣
    def template(self):
        sideLength = self.radius*2 + 1
        result = np.zeros((sideLength, sideLength))
        for i in range(sideLength):
            for j in range(sideLength):
                result[i, j] = self.calc(i-self.radius, j-self.radius)
        all = result.sum()
        return result / all

    def filter(self, image, template):
        arr = np.array(image)
        # print(image)
        height = arr.shape[0]
        width = arr.shape[1]
        newData = np.zeros((height, width))
        print(arr)
        for i in range(self.radius, height - self.radius):
            for j in range(self.radius, width - self.radius):
                t = arr[i - self.radius:i + self.radius +
                        1, j - self.radius: j + self.radius + 1]
                a = np.multiply(t, template)
                newData[i, j] = a.sum()
                # print(newData[i, j])
            # print()
        return newData

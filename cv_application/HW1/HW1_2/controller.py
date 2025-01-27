from email.mime import image
from re import template
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog

import numpy as np
import cv2
import matplotlib.pyplot as plt

from UI import Ui_MainWindow
from gaussian import GaussianBlur
from sobel import Sobel


class MainWindow_controller(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.global_filename = ""
        self.global_image = ""
        self.finish_3_1 = False
        self.finish_3_2 = False
        self.finish_3_3 = False
        self.finish_4_1 = False
        self.finish_4_2 = False
        self.finish_4_3 = False
        # self.global_image2 = ""

    def setup_control(self):
        self.ui.file_button1.clicked.connect(self.open_file)
        self.ui.edge_btn1.clicked.connect(self.edge_GuassianBlur)
        self.ui.edge_btn2.clicked.connect(self.edge_SobelX)
        self.ui.edge_btn3.clicked.connect(self.edge_SobelY)
        self.ui.edge_btn4.clicked.connect(self.edge_Magnitude)
        self.ui.transform_btn1.clicked.connect(self.transfrom_Resize)
        self.ui.transform_btn2.clicked.connect(self.transfrom_Translation)
        self.ui.transform_btn3.clicked.connect(self.transfrom_RotaionScaling)
        self.ui.transform_btn4.clicked.connect(self.transfrom_Shearing)

    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                                                         "Open file",
                                                         "./")
        # print(filename)
        self.global_image = filename
        # self.ui.show_file_path1.setText(filename)
        self.ui.show_file1.setText(filename)

    def cv_imread(self, filepath):
        cv_img = cv2.imdecode(np.fromfile(
            filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv_img

    # 3.1
    def edge_GuassianBlur(self):
        if self.global_image == "":
            return
        radius = 1
        sigma = 0.5 ** 0.5
        GBlur = GaussianBlur(radius, sigma)
        temp = GBlur.template()
        img = self.cv_imread(self.global_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = GBlur.filter(img, temp)
        cv2.imwrite('3_1_Ans.png', img)
        img = cv2.imread('3_1_Ans.png')
        self.finish_3_1 = True
        plt.imshow(img)
        plt.show()

    # 3.2
    def edge_SobelX(self):
        if self.finish_3_1 == False:
            return
        img = cv2.imread("3_1_Ans.png")
        filter = Sobel()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = filter.sobelX(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite('3_2_Ans.png', img)
        img = cv2.imread('3_2_Ans.png')
        #
        self.finish_3_2 = True
        plt.imshow(img)
        plt.show()

    # 3.3

    def edge_SobelY(self):
        if self.finish_3_2 == False:
            return
        img = cv2.imread("3_1_Ans.png")
        filter = Sobel()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = filter.sobelY(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite('3_3_Ans.png', img)
        img = cv2.imread('3_3_Ans.png')
        #
        self.finish_3_3 = True
        plt.imshow(img)
        plt.show()

    # 3.4
    def edge_Magnitude(self):
        if self.finish_3_3 == False:
            return
        filter = Sobel()
        img1 = cv2.imread("3_2_Ans.png")
        img2 = cv2.imread("3_3_Ans.png")
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img = filter.Magnitude(img1, img2)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite('3_4_Ans.png', img)
        img = cv2.imread('3_4_Ans.png')
        plt.imshow(img)
        plt.show()
        pass

    # 4.1
    def transfrom_Resize(self):
        if self.global_image == "":
            return
        img = self.cv_imread(self.global_image)
        print(img.shape)
        rows, cols, _ = img.shape
        new_img = np.zeros((rows, cols, 3), np.uint8)
        img = cv2.resize(img, (int(rows/2), int(cols/2)),
                         interpolation=cv2.INTER_AREA)
        for i in range(int(rows/2)):
            for j in range(int(cols/2)):
                new_img[i][j] = img[i][j]
        cv2.imwrite('4_1_Ans.png', new_img)
        img = cv2.imread('4_1_Ans.png')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
        self.finish_4_1 = True
        plt.imshow(img)
        plt.show()

    # 4.2
    def transfrom_Translation(self):
        if self.finish_4_1 == False:
            return
        img = cv2.imread("4_1_Ans.png")
        rows, cols, _ = img.shape
        for i in range(int(rows/2)):
            for j in range(int(cols/2)):
                img[i + int(rows/2)][j + int(cols/2)] = img[i][j]
        cv2.imwrite('4_2_Ans.png', img)
        img = cv2.imread('4_2_Ans.png')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
        self.finish_4_2 = True
        plt.imshow(img)
        plt.show()

    # 4.3
    def transfrom_RotaionScaling(self):
        if self.finish_4_2 == False:
            return
        img = cv2.imread("4_2_Ans.png")
        rows, cols, _ = img.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 0.5)  # rotate
        img = cv2.warpAffine(img, M, (rows, cols))
        cv2.imwrite('4_3_Ans.png', img)
        img = cv2.imread('4_3_Ans.png')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
        self.finish_4_3 = True
        plt.imshow(img)
        plt.show()

    # 4.4
    def transfrom_Shearing(self):
        if self.finish_4_3 == False:
            return
        img = cv2.imread("4_3_Ans.png")
        rows, cols, _ = img.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [100, 50], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, (cols, rows))
        cv2.imwrite('4_4_Ans.png', img)
        img = cv2.imread('4_4_Ans.png')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
        plt.imshow(img)
        plt.show()
        pass

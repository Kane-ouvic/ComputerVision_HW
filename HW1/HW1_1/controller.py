from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog

import numpy as np
import cv2
import matplotlib.pyplot as plt

from UI import Ui_MainWindow


class MainWindow_controller(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.global_filename = ""
        self.global_image1 = ""
        self.global_image2 = ""

    def setup_control(self):

        self.ui.process_btn1.clicked.connect(self.color_separation)
        self.ui.process_btn2.clicked.connect(self.color_transform)
        self.ui.process_btn3.clicked.connect(self.color_detection)
        self.ui.process_btn4.clicked.connect(self.blending)

        self.ui.smooth_btn1.clicked.connect(self.gaussian_blur)
        self.ui.smooth_btn2.clicked.connect(self.bilateral_filter)
        self.ui.smooth_btn3.clicked.connect(self.median_filter)

        self.ui.file_button1.clicked.connect(self.open_file1)
        self.ui.file_button2.clicked.connect(self.open_file2)

    def open_file1(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                                                         "Open file",
                                                         "./")
        # print(filename)
        self.global_image1 = filename
        # self.ui.show_file_path1.setText(filename)
        self.ui.show_file1.setText(filename)

    def open_file2(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                                                         "Open file",
                                                         "./")
        # print(filename)
        self.global_image2 = filename
        # self.ui.show_file_path2.setText(filename)
        self.ui.show_file2.setText(filename)

    def cv_imread(self, filepath):
        cv_img = cv2.imdecode(np.fromfile(
            filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv_img

    # 1.1

    def show_img(self, img):
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.show()

    def merge_RGBThreeChannel(self, R, G, B):
        img = cv2.merge([B, G, R])
        return img

    def color_separation(self):
        # print(self.global_image1)
        if self.global_image1 == "":
            return
        # image = cv2.imread(self.global_image1)
        image = self.cv_imread(self.global_image1)
        (B, G, R) = cv2.split(image)
        zeros = np.zeros(image.shape[:2], dtype=np.uint8)
        # print("R channel:")
        # self.show_img(self.merge_RGBThreeChannel(R=R, G=zeros, B=zeros))
        # self.show_img(self.merge_RGBThreeChannel(R=zeros, G=G, B=zeros))
        # self.show_img(self.merge_RGBThreeChannel(R=zeros, G=zeros, B=B))

        img_R = self.merge_RGBThreeChannel(R=R, G=zeros, B=zeros)
        img_G = self.merge_RGBThreeChannel(R=zeros, G=G, B=zeros)
        img_B = self.merge_RGBThreeChannel(R=zeros, G=zeros, B=B)
        img_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2RGB)
        img_G = cv2.cvtColor(img_G, cv2.COLOR_BGR2RGB)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        # plot

        image_list = [img_B, img_G, img_R]
        title_list = ['B Channel', 'G Channel', 'R Channel']
        plt.figure(figsize=(20, 5))
        for i in range(len(title_list)):
            plt.subplot(1, 3, i+1)
            plt.imshow(image_list[i])
            plt.axis('off')
            plt.title(title_list[i], size=20)
        plt.tight_layout()
        plt.show()

    # 1.2

    def color_transform(self):
        if self.global_image1 == "":
            return
        image = self.cv_imread(self.global_image1)
        # 0.07*B + 0.72*G + 0.21*R
        img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img2 = self.cv_imread(self.global_image1)  # (R+G+B)/3

        rows, cols, _ = img2.shape
        for i in range(rows):
            for j in range(cols):
                k = img2[i, j]
                rgb = k[0]/3 + k[1]/3 + k[2]/3
                img2[i, j] = (rgb, rgb, rgb)

        # plot
        image_list = [img1, img2]
        title_list = ['1) OpenCV function', '2) Average weighted']
        plt.figure(figsize=(20, 5))
        for i in range(len(title_list)):
            plt.subplot(1, 2, i+1)
            plt.imshow(image_list[i], cmap='gray')
            plt.axis('off')
            plt.title(title_list[i], size=20)
        plt.tight_layout()
        plt.show()

    # 1.3

    def color_detection(self):
        if self.global_image1 == "":
            return
        image = self.cv_imread(self.global_image1)
        hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        lowerb1 = np.array([40, 50, 20])
        upperb1 = np.array((80, 255, 255))
        lowerb2 = np.array([0, 0, 200])
        upperb2 = np.array([180, 20, 255])

        mask1 = cv2.inRange(hsv_img, lowerb1, upperb1)
        mask2 = cv2.inRange(hsv_img, lowerb2, upperb2)
        ship_masked1 = cv2.bitwise_and(image, image, mask=mask1)
        ship_masked2 = cv2.bitwise_and(image, image, mask=mask2)

        image_list = [ship_masked1, ship_masked2]
        title_list = ['Green', 'White']
        plt.figure(figsize=(20, 8))

        for i in range(len(title_list)):
            plt.subplot(1, 2, i+1)
            plt.imshow(image_list[i])
            plt.axis('off')
            plt.title(title_list[i], size=50)
        plt.tight_layout()
        plt.show()

   # 1.4

    def nothing(self, x):
        pass

    def blending(self):
        if self.global_image1 == "" or self.global_image2 == "":
            return
        image1 = self.cv_imread(self.global_image1)
        image2 = self.cv_imread(self.global_image2)
        image = np.zeros((1000, 1000, 3), np.uint8)
        cv2.namedWindow('image')
        cv2.createTrackbar('Blend', 'image', 0, 255, self.nothing)
        image1_rows, image1_cols, _ = image1.shape
        image2_rows, image2_cols, _ = image2.shape

        # if same pixels resize
        if image1_rows != image2_rows or image1_cols != image2_cols:
            image1 = cv2.resize(image1, (256, 256),
                                interpolation=cv2.INTER_AREA)
            image2 = cv2.resize(image2, (256, 256),
                                interpolation=cv2.INTER_AREA)

        while (1):
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            r = cv2.getTrackbarPos('Blend', 'image')
            if (r == -1):
                break
            r = float(r)/255.0
            image = cv2.addWeighted(image1, r, image2, 1.0 - r, 0)
            cv2.imshow('image', image)

        cv2.destroyAllWindows()

    # 2.1
    def gaussian_blur(self):
        if self.global_image1 == "":
            return
        cv2.namedWindow('image')
        cv2.createTrackbar('magnitude', 'image', 0, 10, self.nothing)
        while (1):

            image = self.cv_imread(self.global_image1)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            r = cv2.getTrackbarPos('magnitude', 'image')
            if (r == -1):
                break
            blur = cv2.GaussianBlur(image, (2*r + 1, 2*r + 1), 0)
            image = blur
            cv2.imshow('image', image)

        cv2.destroyAllWindows()

    # 2.2
    def bilateral_filter(self):
        if self.global_image1 == "":
            return
        cv2.namedWindow('image')
        cv2.createTrackbar('magnitude', 'image', 0, 10, self.nothing)
        while (1):
            image = self.cv_imread(self.global_image1)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            r = cv2.getTrackbarPos('magnitude', 'image')
            if (r == -1):
                break
            blur = cv2.bilateralFilter(image, 2*r+1, 90, 90)
            image = blur
            cv2.imshow('image', image)
        cv2.destroyAllWindows()

    # 2.3
    def median_filter(self):
        if self.global_image1 == "":
            return
        cv2.namedWindow('image')
        cv2.createTrackbar('magnitude', 'image', 0, 10, self.nothing)
        while (1):
            image = self.cv_imread(self.global_image1)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            r = cv2.getTrackbarPos('magnitude', 'image')
            if (r == -1):
                break
            blur = cv2.medianBlur(image, 2*r+1)
            image = blur
            cv2.imshow('image', image)
        cv2.destroyAllWindows()

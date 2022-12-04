from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import time

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
        self.global_folder = ""

        self.img1_contour = 0
        self.img2_contour = 0

    def setup_control(self):
        
        self.ui.Load_Folder_btn.clicked.connect(self.open_folder)
        self.ui.Load_imgL_btn.clicked.connect(self.open_file1)
        self.ui.Load_imgR_btn.clicked.connect(self.open_file2)
        self.ui.Ex1_btn1.clicked.connect(self.find_contour)
        self.ui.Ex1_btn2.clicked.connect(self.count_rings)
        self.ui.Ex2_btn1.clicked.connect(self.corner_detection)
        self.ui.Ex2_btn2.clicked.connect(self.intrinsicMT)
        self.ui.Ex2_btn3.clicked.connect(self.extrinsicMT)
        self.ui.Ex2_btn4.clicked.connect(self.distortionMT)
        self.ui.Ex2_btn5.clicked.connect(self.undistoredResult)
        self.ui.Ex3_btn1.clicked.connect(self.showwordBoard)
        self.ui.Ex3_btn2.clicked.connect(self.showwordVert)
        self.ui.Ex4_btn1.clicked.connect(self.stereoMap)

    def open_file1(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                                                         "Open file",
                                                         "./")
        # print(filename)
        self.global_image1 = filename
        # self.ui.show_file_path1.setText(filename)
        self.ui.Load_Data_imgL.setText(filename)

    def open_file2(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                                                         "Open file",
                                                         "./")
        # print(filename)
        self.global_image2 = filename
        # self.ui.show_file_path2.setText(filename)
        self.ui.Load_Data_imgR.setText(filename)

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self,
                                                       "Open folder",
                                                       "./")                 # start path
        print(folder_path)
        self.global_folder = folder_path
        chess_images = glob.glob(self.global_folder + '/*.bmp')
        self.ui.Ex2_nums.setMinimum(1)
        self.ui.Ex2_nums.setMaximum(len(chess_images))
        self.ui.Load_Data_Folder.setText(folder_path)

    def cv_imread(self, filepath):
        cv_img = cv2.imdecode(np.fromfile(
            filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv_img

    # 1.1
    def find_contour(self):
        if self.global_folder == "":
            return
        images_path = glob.glob(self.global_folder + '/*')
        image1 = self.cv_imread(images_path[0])
        image2 = self.cv_imread(images_path[1])

        # img1
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        (thresh, binary) = cv2.threshold(
            gray_image1, 127, 255, cv2.THRESH_BINARY)
        guassian = cv2.GaussianBlur(binary, (11, 11), 0)
        edge_image = cv2.Canny(guassian, 127, 127)
        contours, hierarchy = cv2.findContours(
            edge_image,  cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        image1_copy = image1.copy()
        self.img1_contour = str(int(len(contours) / 4))
        cv2.drawContours(image1_copy, contours, -1, (255, 0, 0), 2)
        imgs = np.hstack([image1,image1_copy])
        cv2.namedWindow('image1')
        cv2.imshow('image1', imgs)

        # img2
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        (thresh, binary) = cv2.threshold(
            gray_image2, 127, 255, cv2.THRESH_BINARY)
        guassian = cv2.GaussianBlur(binary, (11, 11), 0)
        edge_image = cv2.Canny(guassian, 127, 127)
        contours, hierarchy = cv2.findContours(
            edge_image,  cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        image2_copy = image2.copy()
        self.img2_contour = str(int(len(contours) / 4))
        cv2.drawContours(image2_copy, contours, -1, (255, 0, 0), 2)
        imgs = np.hstack([image2,image2_copy])
        cv2.namedWindow('image2')
        cv2.imshow('image2', imgs)
        cv2.waitKey(0)

    # 1.2
    def count_rings(self):
        if self.global_folder == "":
            return
        str1 = 'There are ' + self.img1_contour + ' rings in img1.jpg'
        str2 = 'There are ' + self.img2_contour + ' rings in img2.jpg'
        self.ui.Ex2_text1.setText(str1)
        self.ui.Ex2_text2.setText(str2)
        # print(self.img1_contour)
        # print(self.img2_contour)
        pass

    # 2.1

    def corner_detection(self):
        if self.global_folder == "":
            return
        chess_images = glob.glob(self.global_folder + '/*.bmp')
        for i in range(len(chess_images)):
            # chess_board_image = cv2.imread(chess_images[i])
            chess_board_image = self.cv_imread(chess_images[i])
            gray = cv2.cvtColor(chess_board_image, cv2.COLOR_BGR2GRAY)
            ny = 8
            nx = 11
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret == True:
                cv2.drawChessboardCorners(
                    chess_board_image, (nx, ny), corners, ret)
                result_name = './Ex2_img/board' + str(i+1) + '.bmp'
                cv2.imwrite(result_name, chess_board_image)

        for i in range(len(chess_images)):
            img = cv2.imread('./Ex2_img/board' + str(i+1) + '.bmp')
            plt.imshow(img)
            plt.axis('off')
            plt.ion()
            plt.pause(0.5)
            plt.close()

    # 2.2
    def intrinsicMT(self):
        if self.global_folder == "":
            return
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane

        chess_images = glob.glob(self.global_folder + '/*.bmp')
        # Select any index to grab an image from the list
        for i in range(len(chess_images)):
            image = self.cv_imread(chess_images[i])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(
                    gray, corners, (7, 7), (-1, -1), criteria)
                imgpoints.append(corners2)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (2048, 2048), None, None)
        print(mtx)

    # 2.3

    def extrinsicMT(self):
        if self.global_folder == "":
            return
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane

        chess_images = glob.glob(self.global_folder + '/*.bmp')
        # Select any index to grab an image from the list
        for i in range(len(chess_images)):
            image = self.cv_imread(chess_images[i])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(
                    gray, corners, (7, 7), (-1, -1), criteria)
                imgpoints.append(corners2)
                
        num = self.ui.Ex2_nums.value()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (2048, 2048), None, None)
        R = cv2.Rodrigues(rvecs[num-1])
        ext = np.hstack((R[0], tvecs[num-1]))
        print(ext)

    # 2.4
    def distortionMT(self):
        if self.global_folder == "":
            return
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane

        chess_images = glob.glob(self.global_folder + '/*.bmp')
        # Select any index to grab an image from the list
        for i in range(len(chess_images)):
            image = self.cv_imread(chess_images[i])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(
                    gray, corners, (7, 7), (-1, -1), criteria)
                imgpoints.append(corners2)
        # gray.shape[::-1] = (2048, 2048)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (2048, 2048), None, None)
        print(dist)

    # 2.5
    def undistoredResult(self):
        if self.global_folder == "":
            return
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane

        chess_images = glob.glob(self.global_folder + '/*.bmp')
        # Select any index to grab an image from the list
        for i in range(len(chess_images)):
            image = self.cv_imread(chess_images[i])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(
                    gray, corners, (7, 7), (-1, -1), criteria)
                imgpoints.append(corners2)
    
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (2048, 2048), None, None)
        
        for i in range(len(chess_images)):
            result_name = './Ex2_img/undistorted' + str(i+1) + '.bmp'
            # result_name2 = './Ex2_img/undistorted' + str(i+1) + '.bmp'
            
            img = self.cv_imread(chess_images[i])
            dst = cv2.undistort(img, mtx, dist, None, mtx)
            cv2.imwrite(result_name, dst)
            # cv2.imshow('result', dst)
            # cv2.imwrite('./Ex2_img/result.png', dst)
        
        # show_img1 = img
        # show_img2 = dst
        # imgs = []

        # for i in range(len(chess_images)):
        #     img = self.cv_imread(chess_images[i])
        #     dst = cv2.imread('./Ex2_img/undistorted' + str(i+1) + '.bmp')
        #     rows, cols, _ = img.shape
        #     show_img1 = cv2.resize(img, (int(rows/4), int(cols/4)))
        #     rows, cols, _ = dst.shape
        #     show_img2 = cv2.resize(dst, (int(rows/4), int(cols/4)))
        #     print(i)
        #     imgs.append(np.hstack([show_img1,show_img2]))
        
        # cv2.namedWindow('images')
        # for i in range(len(chess_images)):
        #     cv2.imshow('images',imgs[i])
        #     time.sleep(1)
            
            
        for i in range(len(chess_images)):
            img = self.cv_imread(chess_images[i])
            dst = cv2.imread('./Ex2_img/undistorted' + str(i+1) + '.bmp')
            rows, cols, _ = img.shape
            # show_img1 = cv2.resize(img, (int(rows/4), int(cols/4)))
            rows, cols, _ = dst.shape
            # show_img2 = cv2.resize(dst, (int(rows/4), int(cols/4)))
            imgs = np.hstack([img,dst])
            
            plt.imshow(imgs)
            plt.axis('off')
            plt.ion()
            plt.pause(0.5)
            plt.close()

        
    # 3.1
    
    def draw(self, image, imgpts):
        for i in range(0, len(imgpts), 2):
            image = cv2.line(image, tuple(imgpts[i].ravel()), tuple(
                    imgpts[i+1].ravel()), (0, 0, 255), 5)
        return image


    def transform(self, ch, shift_x, shift_y):
        rows, cols, _ = ch.shape
        for i in range(0, rows):
            for j in range(0, cols):
                ch[i][j][0] = ch[i][j][0] + shift_x
                ch[i][j][1] = ch[i][j][1] + shift_y
        return ch
    
    
    def showword(self, option):
        result_text = self.ui.Ex3_text.text()

        if option == 0:
            fs = cv2.FileStorage('./textlib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
        else:
            fs = cv2.FileStorage('./textlib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
            
        alphabets = []
        if len(result_text) > 0:
            alphabets.append(self.transform(fs.getNode(result_text[0]).mat(), 7, 5))
        if len(result_text) > 1:
            alphabets.append(self.transform(fs.getNode(result_text[1]).mat(), 4, 5))
        if len(result_text) > 2:
            alphabets.append(self.transform(fs.getNode(result_text[2]).mat(), 1, 5))
        if len(result_text) > 3:
            alphabets.append(self.transform(fs.getNode(result_text[3]).mat(), 7, 2))
        if len(result_text) > 4:
            alphabets.append(self.transform(fs.getNode(result_text[4]).mat(), 4, 2))
        if len(result_text) > 5:
            alphabets.append(self.transform(fs.getNode(result_text[5]).mat(), 1, 2))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        axis = []
        for j in range(0, len(alphabets)):
            axis.append(np.float32(alphabets[j]).reshape(-1, 3))
        # axis.append(np.float32(alphabets[1]).reshape(-1, 3))


        objpoints = []
        imgpoints = []

        chess_images = glob.glob(self.global_folder + '/*.bmp')
        for i in range(len(chess_images)):
            image = self.cv_imread(chess_images[i])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                imgpoints.append(corners2)
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, (2048, 2048), None, None)
                for j in range(0, len(alphabets)):

                    imgpts, jac = cv2.projectPoints(
                        axis[j], rvecs[i], tvecs[i], mtx, dist)
                    img = self.draw(image, imgpts)

                # imgpts, jac = cv2.projectPoints(axis[1], rvecs[i], tvecs[i], mtx, dist)
                # img = draw(image, imgpts)

                cv2.imwrite('./Ex3_img/boardword%s.bmp' % (i+1), img)
                img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
        for i in range(len(chess_images)):
            img = cv2.imread('./Ex3_img/boardword' + str(i+1) + '.bmp')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.axis('off')
            plt.ion()
            plt.pause(0.5)
            plt.close()
        pass
    
    def showwordBoard(self):
        if self.global_folder == "":
            return
        self.showword(0)
    
    # 3.2
    def showwordVert(self):
        if self.global_folder == "":
            return
        self.showword(1)
        
    
    # 4.1
    def stereoMap(self):
        
     
        imgL = self.cv_imread(self.global_image1)
        imgR = self.cv_imread(self.global_image2)
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(imgL_gray, imgR_gray).astype(np.float32) / 16.0
        disparity = cv2.resize(disparity, (1400, 950), interpolation=cv2.INTER_AREA)
        focal_len = 4019.284
        baseline = 342.789
        Cx = 279.184
        # print(disparity[700][500])
        cv2.imshow('result', disparity / 16.0)
        
        def draw_circle(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x, ",", y)
                # cv2.rectangle(imgR, (x-3 - 20, y-3), (x+3 - 20, y+3), (0, 0, 255), -1)
                cv2.circle(img=imgR, center=(x-int(disparity[y][x]), y),
                        radius=1, color=(0, 0, 255), thickness=5)
                dist = imgR[y][x] - Cx
                print(dist)
                depth = int(focal_len * baseline / abs(dist))
                print(depth)
        
        while (1):
            cv2.namedWindow('image')
            cv2.setMouseCallback('image', draw_circle)
            cv2.imshow('image', imgL)
            cv2.imshow('image2', imgR)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    
    # 4.2

    ###########################################
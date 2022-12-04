from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog

import numpy as np
import cv2
import matplotlib.pyplot as plt

from UI import Ui_MainWindow

from UI import Ui_MainWindow
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import datetime as dt
import os


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

    def setup_control(self):

        self.ui.Load_Image_btn.clicked.connect(self.open_file1)

        self.ui.Ex5_btn1.clicked.connect(self.show_Train_Images)
        self.ui.Ex5_btn2.clicked.connect(self.show_Distribution)
        self.ui.Ex5_btn3.clicked.connect(self.show_Structure)
        self.ui.Ex5_btn4.clicked.connect(self.show_Comparison)
        self.ui.Ex5_btn5.clicked.connect(self.inference)
        pass

    def open_file1(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                                                         "Open file",
                                                         "./")
        # print(filename)
        self.global_image1 = filename
        # self.ui.show_file_path1.setText(filename)
        self.ui.Image_text.setText(filename)

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
        self.ui.Load_Data_Folder.setText(folder_path)

    def cv_imread(self, filepath):
        cv_img = cv2.imdecode(np.fromfile(
            filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv_img

    def img_loader(self, img_path):
        image = Image.open(img_path)
        img = image.resize((400, 400), Image.ANTIALIAS)
        return img.convert('RGB')

    # 5.1

    def show_Train_Images(self):
        cat_img = cv2.imread('./dataset/inference_dataset/Cat/8043.jpg')
        dog_img = cv2.imread('./dataset/inference_dataset/Dog/12051.jpg')

        cat_img = cv2.resize(cat_img, (224, 224), interpolation=cv2.INTER_AREA)
        dog_img = cv2.resize(dog_img, (224, 224), interpolation=cv2.INTER_AREA)
        cat_img = cv2.cvtColor(cat_img, cv2.COLOR_BGR2RGB)
        dog_img = cv2.cvtColor(dog_img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(7, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(cat_img)
        plt.title('Cat', size=20)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(dog_img)
        plt.title('Dog', size=20)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # 5.2
    def show_Distribution(self):
        img = cv2.imread('./images/distribution.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        pass

    # 5.3
    def show_Structure(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resnet_model = torchvision.models.resnet50(pretrained=False).to(device)
        summary(resnet_model, (3, 224, 224))

    # 5.4
    def show_Comparison(self):
        img = cv2.imread('./images/comparison.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        pass

    # 5.5
    def inference(self):
        if (self.global_image1 == ""):
            return

        def make_prediction(model, filename):
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            img_test_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225])
            ])
            labels = ['Cat', 'Dog']
            img = Image.open(filename)
            img = img_test_transforms(img)
            img = img.unsqueeze(0)
            prediction = model(img.to(device))
            prediction = prediction.argmax()
            img_sh = self.cv_imread(filename)
            img_sh = cv2.cvtColor(img_sh, cv2.COLOR_BGR2RGB)
            plt.imshow(img_sh)
            plt.title("Prediction: " + labels[prediction])
            plt.axis('off')
            plt.show()
            # print(labels[prediction])

        model = torch.load('./models/model.pt')
        make_prediction(model, self.global_image1)

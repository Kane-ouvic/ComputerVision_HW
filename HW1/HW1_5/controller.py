from email.mime import image
from re import template
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import cv2
import matplotlib.pyplot as plt
from UI import Ui_MainWindow
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import datetime as dt


class MainWindow_controller(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.global_filename = ""
        self.global_image = ""
        # self.finish_3_1 = False
        # self.finish_3_2 = False
        # self.finish_3_3 = False
        # self.finish_4_1 = False
        # self.finish_4_2 = False
        # self.finish_4_3 = False
        # self.global_image2 = ""

        # 設定訓練裝置
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device:{self.device}')
        # Hyper-paramesters 超參數
        self.num_epochs = 50
        self.batch_size = 20
        self.learning_rate = 0.01

        # 讀取資料 展示圖片
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']

        # 讀取dataset
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((128, 128)),
                transforms.RandomResizedCrop(
                    size=(128, 128), scale=(0.2, 1.0), ratio=(0.5, 1.1))])

        self.train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform)
        self.test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.transform)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # 讀取資料
        self.dataiter = iter(self.train_loader)
        self.images, self.labels = self.dataiter.next()

        # 讀取模型
        self.model = torchvision.models.vgg19().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate)

    def setup_control(self):
        self.ui.file_button1.clicked.connect(self.open_file)
        self.ui.train_btn1.clicked.connect(self.show_Train_Images)
        self.ui.train_btn2.clicked.connect(self.show_Model_Structure)
        self.ui.train_btn3.clicked.connect(self.show_Data_Augmentation)
        self.ui.train_btn4.clicked.connect(self.show_AccuracyAndLoss)
        self.ui.train_btn5.clicked.connect(self.inference)
        self.ui.train_btn6.clicked.connect(self.startTrain)

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

    def img_loader(self, img_path):
        image = Image.open(img_path)
        img = image.resize((400, 400), Image.ANTIALIAS)
        return img.convert('RGB')

    # 5.1 展示圖片
    def show_Train_Images(self):
        # tranform2 = transforms.Compose([transforms.RandomHorizontalFlip(0.5)])
        tranform1 = transforms.Compose([transforms.Resize((32, 32))])

        train_dataset1 = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform)

        train_loader1 = torch.utils.data.DataLoader(
            train_dataset1, batch_size=self.batch_size, shuffle=True)

        # 讀取資料
        dataiter = iter(train_loader1)
        images, labels = dataiter.next()

        plt.figure(figsize=(7, 7))
        for i in range(9):
            plt.subplot(3, 3, i+1)
            images[i] = images[i] / 2 + 0.5     # unnormalize
            npimg = images[i].numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.axis('off')
            plt.title(self.classes[labels[i]], size=10)
        plt.tight_layout()
        plt.show()

    # 5.2 讀取模型
    def show_Model_Structure(self):
        summary(self.model, (3, 32, 32))

    # 5.3
    def show_Data_Augmentation(self):
        if self.global_image == "":
            return
        img = self.img_loader(self.global_image)
        tranform = transforms.Compose([transforms.RandomHorizontalFlip(0.5)])
        images1 = tranform(img)

        tranform = transforms.Compose([transforms.RandomRotation(10)])
        images2 = tranform(img)
        tranform = transforms.Compose([transforms.RandomResizedCrop(
            size=(200, 200), scale=(0.2, 1.0), ratio=(0.5, 1.1))])
        images3 = tranform(img)

        image_list = [images1, images2, images3]
        plt.figure(figsize=(7, 7))
        for i in range(len(image_list)):
            plt.subplot(1, 3, i+1)
            plt.imshow(image_list[i])
            plt.axis('off')
        plt.subplots_adjust(wspace=2)
        plt.tight_layout()
        plt.show()

    # 5.4
    def show_AccuracyAndLoss(self):
        # img = self.cv_imread("accuracy_loss.png")
        img = self.img_loader(r"accuracy_loss.png")
        img.show()
        pass

    # 5.5
    def inference(self):
        if (self.global_image == ""):
            return
        self.model = torch.load('./models/model.pt')
        img = self.cv_imread(self.global_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_ = self.transform(img).unsqueeze(0)
        img_ = img_.to(self.device)

        outputs = self.model(img_)
        confidence, predicted = torch.max(outputs, 1)
        percentage = torch.nn.functional.softmax(outputs, dim=1)[0]*100
        perc = percentage[int(predicted)].item()
        # print(perc)
        # print(percentage)
        # print("======================")
        # print(confidence)
        # print(predicted)
        # print(type(confidence))
        # print(type(predicted))
        test1 = confidence.cpu().detach().numpy()
        # print(str(test1[0]/100))
        test2 = predicted.cpu().detach().numpy()
        # print(self.classes[test2[0]])
        plt.imshow(img)
        plt.title("Confidence = " +
                  str(round(perc/100, 2)) + "\nPrediction Label: " + self.classes[test2[0]])
        plt.show()

    # start training
    def startTrain(self):
        Loss_list = []
        Accuracy_list_train = []
        Accuracy_list_test = []
        n_total_steps = len(self.train_loader)
        t1 = dt.datetime.now()
        print(f'Training started at {t1}')
        for epoch in range(self.num_epochs):
            # loss_rate = 0.0
            # accuracy_rate = 0.0
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i+1) % 10 == 0:
                    print(
                        f'Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{n_total_steps}], loss: {loss.item():.4f}')

                if (i+1) % n_total_steps == 0:
                    Loss_list.append(loss.item())

            # 測試訓練結果 -- Training Data
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                for i in range(self.batch_size):
                    label = labels[i]
                    pred = predicted[i]
                    if label == pred:
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1
            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network(Training Data): {acc} %')
            Accuracy_list_train.append(acc)

            # 測試訓練結果 -- Testing Data
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                for i in range(self.batch_size):
                    label = labels[i]
                    pred = predicted[i]
                    if label == pred:
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1
            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network(Testing Data): {acc} %')
            Accuracy_list_test.append(acc)

            # print(Loss_list)
            # print(Accuracy_list_train)
            # print(Accuracy_list_test)

        t2 = dt.datetime.now()
        print(f'Finished Training at {t2}')
        print(f'Training time :  {t2-t1}')

        # 繪製 Accuracy and Epoch
        x1 = range(0, self.num_epochs)
        x2 = range(0, self.num_epochs)
        y1_1 = Accuracy_list_train
        y1_2 = Accuracy_list_test
        y2 = Loss_list
        plt.subplot(2, 1, 1)
        plt.plot(x1, y1_1, '-')
        plt.plot(x1, y1_2, '-')
        plt.title('Accuracy')
        plt.ylabel('%')
        plt.legend(['Training', 'Testing'])
        #
        plt.subplot(2, 1, 2)
        plt.plot(x2, y2, '-')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig("accuracy_loss.png")
        plt.show()

        # 儲存模型
        torch.save(self.model, './models/model.pt')

from email.mime import image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import datetime as dt


def img_loader(img_path):
    image = Image.open(img_path)
    img = image.resize((400, 400), Image.ANTIALIAS)
    return img.convert('RGB')


# 設定訓練裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')

# Hyper-paramesters 超參數
num_epochs = 5
batch_size = 10
learning_rate = 0.001

# 讀取dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)


# 讀取資料 展示圖片
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

dataiter = iter(train_loader)
images, labels = dataiter.next()

plt.figure(figsize=(7, 7))
for i in range(9):
    plt.subplot(3, 3, i+1)
    images[i] = images[i] / 2 + 0.5     # unnormalize
    npimg = images[i].numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.title(classes[labels[i]], size=10)
plt.tight_layout()
plt.show()


# 讀取模型
model = torchvision.models.vgg19().to(device)
summary(model, (3, 32, 32))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 資料增強
# transform1 = transforms.Compose(
#     [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# img = Image.open("13.png")
# print(img)
# print("testtttttttttttttttttttttttttttt")
# img = transform1(img)
# print(img)
# img.save("test_R180.jpg")
img = img_loader(r"13.png")
img.show()
tranform = transforms.Compose([transforms.RandomHorizontalFlip(0.5)])
images1 = tranform(img)

tranform = transforms.Compose([transforms.RandomRotation(10)])
images2 = tranform(img)
# tranform = transforms.Compose(
#     [transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2))])
tranform = transforms.Compose([transforms.RandomResizedCrop(
    size=(200, 200), scale=(0.2, 1.0), ratio=(0.5, 1.1))])
images3 = tranform(img)

image_list = [images1, images2, images3]
plt.figure(figsize=(7, 7))
for i in range(len(image_list)):
    plt.subplot(1, 3, i+1)
    plt.imshow(image_list[i])
    plt.axis('off')
    # plt.title(classes[labels[i]], size=10)
plt.subplots_adjust(wspace=2)
plt.tight_layout()
plt.show()


# 開始訓練
Loss_list = []
Accuracy_list_train = []
Accuracy_list_test = []
n_total_steps = len(train_loader)
t1 = dt.datetime.now()
print(f'Training started at {t1}')
for epoch in range(num_epochs):
    # loss_rate = 0.0
    # accuracy_rate = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 5000 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], loss: {loss.item():.4f}')
            Loss_list.append(loss.item())

    # 測試訓練結果 -- Training Data
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
    Accuracy_list_train.append(acc)

    # 測試訓練結果 -- Testing Data
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
    Accuracy_list_test.append(acc)

    print(Loss_list)
    print(Accuracy_list_train)
    print(Accuracy_list_test)
    # for i in range(10):
    #     acc = 100.0 * n_class_correct[i] / n_class_samples[i]
    #     print(f'Accuracy of {classes[i]}: {acc} %')

    ##

t2 = dt.datetime.now()
print(f'Finished Training at {t2}')
print(f'Training time :  {t2-t1}')

# 繪製 Accuracy and Epoch
x1 = range(0, num_epochs)
x2 = range(0, num_epochs)
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


# 驗證訓練結果
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

# 儲存模型
torch.save(model, './models/model.pt')


# 載入模型使用
# model = torch.load('./models/model.pt')
# images = images.to(device)
# outputs = model(images)
# confidence, predicted = torch.max(outputs, 1)

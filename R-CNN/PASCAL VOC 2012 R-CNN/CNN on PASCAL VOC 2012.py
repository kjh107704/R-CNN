# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from __future__ import print_function, division
import json 
import numpy as np
import cv2
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
import time
import os
import copy
import sys

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree

import random

from shutil import copyfile


# %%
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# %% [markdown]
# # training dataset 만들기
# 
# PASCAL VOC 2012(class 20개) 데이터 사용함.
# 
# Annotation이 존재하는 모든 이미지에서 object를 crop한 뒤,
# 각 class 별로 train: 최대 800개, val: 100개, test: 100개 이미지를 가지도록 split 함.

# %%
PASCAL_PATH = './VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'

IMAGE_PATH = 'JPEGImages/'
ANNOTATION_PATH = 'Annotations/'

DATASET_PATH = './Dataset/'


# %%
def InitializeNumOfImg():
    for i in range(20):
        num_of_img[i] = 0


# %%
num_of_img = {}
InitializeNumOfImg()


# %%
class_name = ["aeroplane",
              "bicycle",
              "bird",
              "boat",
              "bottle",
              "bus",
              "car",
              "cat",
              "chair",
              "cow",
              "diningtable",
              "dog",
              "horse",
              "motorbike",
              "person",
              "pottedplant",
              "sheep",
              "sofa",
              "train",
              "tvmonitor"
             ]

# %% [markdown]
# ### 이미지를 지정된 폴더에 저장
# 
# - train img: DATASET_PATH/train/class번호/ 폴더 내부에 이미지 저장
#     
# - val img: DATASET_PATH/val/class번호/ 폴더 내부에 이미지 저장
#     
# - test img: DATASET_PATH/test/class번호/ 폴더 내부에 이미지 저장
#     
# - base img (전체 데이터 편집 시 사용): DATASET_PATH/base/class번호/ 폴더 내부에 이미지 저장

# %%
def custom_imsave(img, label, mode = 'base'):
    
    if mode == 'train' or mode == 'trainval':
        path = DATASET_PATH + 'train/' + str(label) + '/'
    elif mode == 'val':
        path = DATASET_PATH + 'val/' + str(label) + '/'
    elif mode == 'test':
        path = DATASET_PATH + 'test/' + str(label) + '/'
    elif mode == 'base':
        path = DATASET_PATH + 'base/' + str(label) + '/'

    if not os.path.exists(path):
        os.makedirs(path)
        
    cv2.imwrite(path+str(num_of_img[label])+'.jpg', img)
    num_of_img[label] += 1

# %% [markdown]
# ### Annotation이 존재하는 모든 이미지를 crop하여 class별로 저장

# %%
def make_base_dataset():
    mypath = PASCAL_PATH+'/Annotations'
    img_list = [f.split('.')[0] for f in os.listdir(mypath) if f.endswith('.xml')]
    print(f'total image: {len(img_list)}')
    
    for index, img_name in enumerate(img_list):
        printProgressBar(index, len(img_list), prefix='Progress', suffix='Complete', length=50)
        tmp_img = cv2.imread(PASCAL_PATH+IMAGE_PATH+'/'+img_name+'.jpg')
        imout = tmp_img.copy()

        gtvalues = []

        img_xml = open(PASCAL_PATH+ANNOTATION_PATH+'/'+img_name+'.xml')
        tree = Et.parse(img_xml)
        root = tree.getroot()

        objects = root.findall("object")

        # Annotation 기준으로 object 추출
        for _object in objects:
            name = _object.find("name").text
            bndbox = _object.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            
            timage = imout[ymin:ymax, xmin:xmax]
            # 정의된 class에 존재하는 object일 경우 이미지 crop 및 저장
            if name in class_name:
                class_num = class_name.index(name)
                custom_imsave(timage, class_num)
    printProgressBar(len(img_list), len(img_list), prefix='Progress', suffix='Complete', length=50)


# %%
make_base_dataset()


# %%
def split_data_into_train_val_test():
    path_list = [DATASET_PATH+'train/', DATASET_PATH+'val/', DATASET_PATH+'test/']
    
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)
            for i in num_of_img:
                if not os.path.exists(os.path.join(path,str(i))):
                    os.makedirs(os.path.join(path,str(i)))
    
    for i in num_of_img:
        print(f'class {i} has {num_of_img[i]} items')
        class_path = os.path.join(DATASET_PATH+'base/',str(i))
        img_list = [f for f in os.listdir(class_path)]
        random.shuffle(img_list)
        
        for index, img_name in enumerate(img_list):
            if index < 100:
                copyfile(os.path.join(class_path,img_name),os.path.join(path_list[1],str(i),img_name))
            elif index < 200:
                copyfile(os.path.join(class_path,img_name),os.path.join(path_list[2],str(i),img_name))
            elif index < 1000:
                copyfile(os.path.join(class_path,img_name),os.path.join(path_list[0],str(i),img_name))
        


# %%
split_data_into_train_val_test()

# %% [markdown]
# # VGG16 모델 이용하여 CNN 적용하기
# %% [markdown]
# ## 데이터 가져오기

# %%
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


# %%
image_datasets = {x: datasets.ImageFolder(os.path.join(DATASET_PATH, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## 데이터 확인하기

# %%
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.


# %%
# 학습 데이터의 배치를 얻습니다.
inputs, classes = next(iter(dataloaders['train']))

# 배치로부터 격자 형태의 이미지를 만듭니다.
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

# %% [markdown]
# ## 모델 학습하기

# %%
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for index, (inputs, labels) in enumerate(dataloaders[phase]):
                printProgressBar (index, len(dataloaders[phase]), prefix='Progress', suffix='Complete', length=50)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            printProgressBar (len(dataloaders[phase]), len(dataloaders[phase]), prefix='Progress', suffix='Complete', length=50)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    return model


# %%
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# %%
model = models.vgg16(pretrained=False)
model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names))
model = model.to(device)
criterion = nn.CrossEntropyLoss()

# 모든 매개변수들이 최적화되었는지 관찰
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 7 에폭마다 0.1씩 학습율 감소
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# %%
model = train_model(model, 
                       criterion, 
                       optimizer_ft, 
                       exp_lr_scheduler,
                       num_epochs=25)

# %% [markdown]
# ## 모델 평가하기

# %%
def test_model(model):
    since = time.time()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for index, (inputs, labels) in enumerate(dataloaders['test']):
            printProgressBar (index, len(dataloaders['test']), prefix='Progress', suffix='Complete', length=50)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            
            total += labels.size(0)
            correct += torch.sum(preds == labels.data)
            
    printProgressBar (len(dataloaders['test']), len(dataloaders['test']), prefix='Progress', suffix='Complete', length=50)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Test Acc: {:4f}'.format(correct.double() / dataset_sizes['test']))


# %%
test_model(model)

# %% [markdown]
# ## 모델 저장하기

# %%
torch.save(model.state_dict(), './TrainedModel/PASCAL2012CNN')



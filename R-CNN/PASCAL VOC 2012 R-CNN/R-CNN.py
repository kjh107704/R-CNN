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

import urllib.request as urllib2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_PATH = './Dataset/'


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


# %%
model = models.vgg16(pretrained=False)
model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names))
model = model.to(device)


# %%
model.load_state_dict(torch.load('./TrainedModel/PretrainedTrue'))
model.eval()


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
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            #labels = labels.to(device)

            outputs = model(inputs)
            for i in outputs:
                print(i)
            _, preds = torch.max(outputs, 1)
            
            print(preds.size)
            
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
visualize_model(model)


# %%
def cv2_selective_search(img, searchMethod='f'):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    
    if searchMethod == 'f':
        ss.switchToSelectiveSearchFast()
    elif searchMethod == 'q':
        ss.switchToSelectiveSearchQuality()
        
    regions = ss.process()
    
    return regions


# %%
def draw_bounding_box(bounding_box, img):
    tmp_img = img.copy()

    dim = np.array(bounding_box).ndim
    
    if dim == 2:
        for x, y, w, h in bounding_box:
            cv2.rectangle(tmp_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    elif dim == 3:
        for bb in bounding_box:
            for x, y, w, h in bb:
                cv2.rectangle(tmp_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    return cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)


# %%
IMG_URL = "https://www.telegraph.co.uk/content/dam/Travel/2018/January/white-plane-sky.jpg?imwidth=450"


# %%

img = Image.open(urllib2.urlopen(IMG_URL))

opencvImg = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

plt.imshow(cv2.cvtColor(opencvImg, cv2.COLOR_BGR2RGB))


# %%
def get_predict(model, img):
    model.eval()
    
    with torch.no_grad():
        inputs = img.to(device)
        inputs = inputs.unsqueeze(0)
        outputs = model(inputs)
        #print(outputs)
        _, preds = torch.max(outputs, 1)
        print(_)
        print(preds)
        return(_,class_name[preds])


# %%
_img = img
_img = data_transforms['test'](transforms.ToPILImage()(np.asarray(_img)))
get_predict(model, _img)


# %%
img = Image.open(DATASET_PATH+'sample/bicycle_2.jpg')

opencvImg = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

plt.imshow(cv2.cvtColor(opencvImg, cv2.COLOR_BGR2RGB))


# %%
_img = img
_img = data_transforms['test'](transforms.ToPILImage()(np.asarray(_img)))
get_predict(model, _img)


# %%
SS_BB = cv2_selective_search(opencvImg)

print(f'num of all regions by ss: {len(SS_BB)}')
plt.imshow(draw_bounding_box(SS_BB, opencvImg))


# %%
def get_candidate_bounding_box(SS_BB):
    bounding_box = []
    
    for index, ss_bb in enumerate(SS_BB):
        if index < 100:
            bounding_box.append(ss_bb)
    return bounding_box


# %%
bounding_box = get_candidate_bounding_box(SS_BB)


# %%
scores = []
labels = []

for index, (x, y, w, h) in enumerate(bounding_box):
    area = (x,y,x+w,y+h)
    timage = img.crop(area)
    timage = data_transforms['test'](transforms.ToPILImage()(np.asarray(timage)))
    (score, label) = get_predict(model, timage)
    
    scores.append(score)
    labels.append(label)
    
keep = torchvision.ops.nms(torch.tensor(bounding_box).float(), torch.tensor(scores).float(), 0.5)

for index, (result, bbox, label) in enumerate(zip(keep, bounding_box, labels)):
    if result == True:
        x,y,w,h, = bbox
        opencvImg = cv2.rectangle(opencvImg, (x, y,), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(opencvImg, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
    
#cv2.imsave(opencvImg)
plt.imsave("result.jpg",cv2.cvtColor(opencvImg, cv2.COLOR_BGR2RGB))


# %%




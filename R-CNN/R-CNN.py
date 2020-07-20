# %%
import os,cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import json

# %%
BASE_PATH = "R-CNN/"
PATH = "Images"
ANNOT = "Airplanes_Annotations"

# %%
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
# %%
# calculate IOU(Interscetion Over Union)
# for more information [Intersection over Union (IoU) for object detection](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)

def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
# %%
train_images=[]
train_labels=[]

for e,i in enumerate(os.listdir(ANNOT)):
    try:
        if i.startswith("airplane"):
            filename = i.split(".")[0]+".jpg"
            print(e,filename)
            image = cv2.imread(os.path.join(PATH,filename))
            df = pd.read_csv(os.path.join(ANNOT,i))
            gtvalues=[]
            for row in df.iterrows():
                x1 = int(row[1][0].split(" ")[0])
                y1 = int(row[1][0].split(" ")[1])
                x2 = int(row[1][0].split(" ")[2])
                y2 = int(row[1][0].split(" ")[3])
                gtvalues.append({"x1":x1,"x2":x2,"y1":y1,"y2":y2})
            ss.setBaseImage(image)
            ss.switchToSelectiveSearchFast()
            ssresults = ss.process()
            imout = image.copy()
            counter = 0
            falsecounter = 0
            flag = 0
            fflag = 0
            bflag = 0
            for e,result in enumerate(ssresults):
                if e < 2000 and flag == 0:
                    for gtval in gtvalues:
                        x,y,w,h = result
                        iou = get_iou(gtval,{"x1":x,"x2":x+w,"y1":y,"y2":y+h})
                        # 최대 30개의 positive sample(airplane) 저장
                        if counter < 30:
                            if iou > 0.70:
                                timage = imout[y:y+h,x:x+w]
                                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(1)
                                counter += 1
                        else :
                            fflag =1
                        # 최대 30개의 negative sampel(background) 저장
                        if falsecounter <30:
                            if iou < 0.3:
                                timage = imout[y:y+h,x:x+w]
                                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(0)
                                falsecounter += 1
                        else :
                            bflag = 1
                    if fflag == 1 and bflag == 1:
                        print("inside")
                        flag = 1
    except Exception as e:
        print(e)
        print("error in "+filename)
        continue

# %%
print(train_images[0].shape)
# %%
class AirplaneAndBackgroundDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, index):
        img = self.x_data[index]
        label = self.y_data[index]
        return img, label
# %%
X_new = np.array(train_images)
X_tensor = transforms.ToTensor
print(X_tensor.size())
Y_new = np.array(train_labels)
# %%
train_dataSet = AirplaneAndBackgroundDataset(X_new,Y_new)
train_dataloader = DataLoader(dataset=train_dataSet,batch_size=2,shuffle=True)
# %%
vgg = models.vgg16(pretrained=True)

for param in vgg.features.parameters():
    param.requires_grad = False

vgg.classifier[6].out_features = 2

optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.0001, momentum=0.5)
# %%

def fit(epoch, model, data_loader, phase='training', volatile=False):
    is_cuda = torch.cuda.is_available()
    optimizer = optim.Adam(model.parameters())
    
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile=True
    
    running_loss = 0.0
    running_correct = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)

        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)

        running_loss += F.nll_loss(output, target, size_average=False).data[0]
        preds = output.data.max(dim=1, keepdim=True)

        if phase == 'training':
            loss.backward()
            optimizer.step()

        loss = running_loss/len(data_loader.dataset)
        accuracy = 100. * running_correct/len(data_loader.dataset)

        print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
        return loss, accuracy
# %%

train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []

for epoch in range(1,20):
    epoch_loss, epoch_accuracy = fit(epoch, vgg, train_dataloader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, model, train_dataloader, phase='training')

    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
from __future__ import print_function, division

import matplotlib.pyplot as plt
import random
import numpy as np
import cv2

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import os

import urllib.request as urllib2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
import time
import copy
import sys

classes = ('aeroplane','bicycle','diningtable',
           'dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor',
           'bird','boat','bottle','bus','car','cat','chair','cow')

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

class SAVE_IMAGE:
    
    def __init__(self, ncols = 0, nrows = 0, figTitle=""):
        
        if ncols == 0 or nrows == 0:
            raise ValueError("ncols and nrows must be initialize")
        
        dpi = 80
        height, width, depth = CV2_IMG.shape
        figsize = width / float(dpi) * ncols , height / float(dpi) * nrows
        self.fig, self.ax = plt.subplots(ncols = ncols, nrows = nrows, figsize=figsize)
        self.ncols = ncols
        self.nrows = nrows
        
        if figTitle is not "":
            self.fig.suptitle(figTitle, fontsize=20)
        self.ccols = 0
        self.crows = 0
        
    def addImage(self, img, title = ""):
        
        if self.nrows == 1:
            if self.ncols == 1:
                self.ax.imshow(img)
                self.ax.set_title(title, fontsize=15)
            else:
                self.ax[self.ccols].imshow(img)
                self.ax[self.ccols].set_title(title, fontsize=15)
        else:
            self.ax[self.crows][self.ccols].imshow(img)
            self.ax[self.crows][self.ccols].set_title(title, fontsize=15)

        if self.ccols+1 == self.ncols:
            self.crows = self.crows + 1
            self.ccols = 0
        else:
            self.ccols = self.ccols + 1
            
    def showImage(self):
        plt.show()
        
    def saveImage(self, save_path, save_title):
        plt.savefig(save_path+save_title+'.png', bbox_inches='tight')


def GenerateRandomColor(num_of_class):
    color = []

    while len(color) < num_of_class:
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        rgb = [r,g,b]
        color.append(rgb)
    
    return color

def CheckDirExists(PATH, DIR):
    if not os.path.exists(PATH+DIR):
        os.makedirs(PATH+DIR)


def SaveOriginalImage(img):
    
    save_image = SAVE_IMAGE(nrows = 1, ncols = 1, figTitle="base image")
    
    save_image.addImage(img)
    save_image.saveImage(RESULT_PATH+RESULT_DIR, "base_image")


def GetHeatmap(img_list, height, width, title = "", figSet = False, fig = [0, 0]):
    _title = "_color_heatmap"
    heatmaps = []
    
    if figSet:
        save_image = SAVE_IMAGE(nrows = fig[0], ncols = fig[1], figTitle=title+_title)
    else: 
        save_image = SAVE_IMAGE(nrows = 1, ncols = len(img_list), figTitle=title+_title)
    
    for index, img in enumerate(img_list):
        heatmap = cv2.applyColorMap(cv2.resize(img, (width, height)), cv2.COLORMAP_JET)
        heatmaps.append(heatmap)
        tmp_img = heatmap*0.6 + CV2_IMG*0.4
        save_image.addImage(cv2.cvtColor(np.float32(tmp_img).astype('uint8'), cv2.COLOR_BGR2RGB))
        
    save_image.saveImage(RESULT_PATH+RESULT_DIR, title+_title)
    
    return heatmaps


# orig_img에서 (R, G, B) 세 가지 채널의 정보 중 특정 채널의 정보만 남겨서 넘김
def GetChannelImage(orig_img, channel):
    channel = channel.upper()
    channel_img = orig_img.copy()
    if channel == 'R':
        channel_img[:, :, 0] = 0
        channel_img[:, :, 1] = 0
    elif channel == 'G':
        channel_img[:, :, 0] = 0
        channel_img[:, :, 2] = 0
    elif channel == 'B':
        channel_img[:, :, 1] = 0
        channel_img[:, :, 2] = 0

    return channel_img


# color image를 gray scale로 바꾼 후, threshold를 적용함
# threshold는 고정 값으로 mean(min, max)
def GetGrayscaleImageWithThreshold(orig_img):
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    
    min_val = np.min(gray_img)
    max_val = np.max(gray_img)
    threshold = (min_val + max_val) / 2
    
    ret, gray_img = cv2.threshold(gray_img, threshold, 1, cv2.THRESH_BINARY)    

    return gray_img


# grayscale_mask 에서 1인 부분만 orig_img를 보여줌. 0인 부분은 검정색으로 보임
def GetMaskedImage(orig_img, gray_map):
    mask = cv2.cvtColor(gray_map, cv2.COLOR_GRAY2BGR)
    
    maskedRegion = np.where(mask == 1, orig_img, 0)
    
    return cv2.cvtColor(maskedRegion, cv2.COLOR_BGR2RGB)


def GetGrayscaleHeatmap(heatmaps, title = "", figSet = False, fig = [0, 0]):
    _title = "_grayscale_heatmap"
    
    result = []
    
    for index, heatmap in enumerate(heatmaps):
        # heatmap에서 R channel만 뽑아냄
        tmp = GetChannelImage(heatmap, 'r')
        # grayscale로 변환 후 threshold 적용
        result.append(GetGrayscaleImageWithThreshold(tmp))
    
    if figSet:
        save_image = SAVE_IMAGE(nrows = fig[0], ncols = fig[1], figTitle=title+_title)
    else:
        save_image = SAVE_IMAGE(nrows = 1, ncols = len(result), figTitle=title+_title)
    
    for index, graymap in enumerate(result):
        save_image.addImage(GetMaskedImage(CV2_IMG, graymap))
    
    save_image.saveImage(RESULT_PATH+RESULT_DIR, title+_title)
        
    return result


def GetContours(img_binary):
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours


def GetBBox(img_binary):
    bb = []
    
    contours = GetContours(img_binary)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bb.append([x, y, w, h])
        
    return bb


def DrawBBox(bounding_box, img):
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


def DrawContourAndBBox(img_binary, img):
    contours = GetContours(img_binary)
    
    tmp_img = img.copy()
    
    # draw contours - red
    for cnt in contours:
        cv2.drawContours(tmp_img, [cnt], 0, (0,0,255),3)
    
    # draw bounding box - green
    bb = GetBBox(img_binary)
    
    for x, y, w, h in bb:
        cv2.rectangle(tmp_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
    return cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)

def CompareContourAndBBox(heatmaps, title = "", figSet = False, fig = [0, 0]):
    _title = "_contour"
    
    if figSet:
        save_image = SAVE_IMAGE(nrows = fig[0], ncols = fig[1], figTitle=title+_title)
    else:
        save_image = SAVE_IMAGE(nrows = 1, ncols = len(heatmaps), figTitle=title+_title)
    for index, heatmap in enumerate(heatmaps):
        save_image.addImage(DrawContourAndBBox(heatmap, CV2_IMG))
    save_image.saveImage(RESULT_PATH+RESULT_DIR, title+_title)


def GetIOU(_bb1, _bb2, changeScale = False, basedOnCAM = False):
    # _bb2 == cam_bb
    if changeScale:
        # _bb1, _bb2 = [x, y, w, h]
        if len(_bb1) == 4 and len(_bb2) == 4:
            bb1 = {'x1':_bb1[0], 'y1':_bb1[1], 'x2':_bb1[0]+_bb1[2], 'y2':_bb1[1]+_bb1[3]}
            bb2 = {'x1':_bb2[0], 'y1':_bb2[1], 'x2':_bb2[0]+_bb2[2], 'y2':_bb2[1]+_bb2[3]}
        else:
            exit(0)
    else:
        # _bb1, _bb2 = ['x1':x1, 'x2':x2, 'y1':y1, 'y2':y2]
        x1, y1, x2, y2 = _bb1
        bb1 = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        x1, y1, x2, y2 = _bb2
        bb2 = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    
    
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    
    if basedOnCAM:
        # cam_bb 기준 iou
        iou = intersection_area / float(bb2_area)
    else:
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def isExist(bounding_box, bb):
    for _bb in bounding_box:
        if np.array_equal(_bb, bb):
            return True
    return False


def GetCandidateBBox(FM_BB, CAM_BB):
    # FM_BB dim = 2
    # CAM_BB dim = 2
    
    bounding_box = []

    for fm_bb in FM_BB:
        for cam_bb in CAM_BB:
            iou = GetIOU(fm_bb, cam_bb, changeScale = True, basedOnCAM=True)
            if iou > 0.7:
                if not isExist(bounding_box, fm_bb):
                    bounding_box.append(fm_bb)
                
    return bounding_box


def NMS(bounding_box, probs):
    
    bbox = []

    for x, y, w, h in bounding_box:
        bbox.append([x,y, x+w, y+h])
    
    
    _opencvImg = CV2_IMG.copy()
    bbox = torch.as_tensor(bbox).float()
    probs = torch.as_tensor(probs)
    for c in range(len(classes)):
        
        _cnt = 0
        
        # threshold 적용
        
        prob = probs[:, c].clone()
        
        m = nn.Threshold(0.2, 0)
        
        prob = m(prob)
        
        order = torch.argsort(prob, descending=True)
        
        for i in range(len(order)):
           
            bbox_max = bbox[order[i]]
            for j in range(i+1, len(order)):
                bbox_cur = bbox[order[j]]
                
                if GetIOU(bbox_max, bbox_cur) > 0.5:
                    prob[order[j]] = 0
        
        
        probs[:, c] = prob
        
    return probs
    return 


def get_predict(model, img):
    model.eval()
    
    with torch.no_grad():
        inputs = img.to(device)
        inputs = inputs.unsqueeze(0)
        outputs = model(inputs)
        softmax = nn.Softmax(dim=1)
        outputs = softmax(outputs)
        return outputs


def DrawResultByClass(bounding_box, probs, fig = [5, 4]):
    
    _opencvImg = CV2_IMG.copy()
    save_image = SAVE_IMAGE(nrows = fig[0], ncols = fig[1], figTitle="")
    
    for i in range(20):
        row = int(i / 5)
        col = i % 5
        
        _opencvImg = CV2_IMG.copy()
        
        draw = 0
        
        for cnt in range(len(bounding_box)):
            
            cls_idx = torch.argsort(probs[cnt, :], descending=True)[0]
            if cls_idx == i:
                if probs[cnt][cls_idx] > 0:
                    draw += 1
                    x,y,w,h = bounding_box[cnt]
                    _opencvImg = cv2.rectangle(_opencvImg, (x, y,), (x+w, y+h), color[cls_idx], 2)
                    text = '{} ({:.3f})'.format(classes[cls_idx], probs[cnt][cls_idx])
                    cv2.putText(_opencvImg, text, (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, color[cls_idx], 2)
        
        title = classes[i] + ": "+str(draw)
        save_image.addImage(cv2.cvtColor(_opencvImg, cv2.COLOR_BGR2RGB), title=title)

    save_image.saveImage(RESULT_PATH+RESULT_DIR, "draw_result_by_class")


def cv2_selective_search(img, searchMethod='f'):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    
    if searchMethod == 'f':
        ss.switchToSelectiveSearchFast()
    elif searchMethod == 'q':
        ss.switchToSelectiveSearchQuality()
        
    regions = ss.process()
    
    return regions


def DrawResult(bounding_box, probs):
    
    draw = 0
    _opencvImg = CV2_IMG.copy()
    
    
    
    for cnt in range(len(bounding_box)):
        
        cls_idx = torch.argsort(probs[cnt, :], descending=True)[0]
        
        if probs[cnt][cls_idx] > 0:
            
            draw += 1
            x,y,w,h = bounding_box[cnt]
            _opencvImg = cv2.rectangle(_opencvImg, (x, y,), (x+w, y+h), color[cls_idx], 2)
            text = '{} ({:.3f})'.format(classes[cls_idx], probs[cnt][cls_idx])
            cv2.putText(_opencvImg, text, (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, color[cls_idx], 2) 
            
    
    title = 'final bbox: {}'.format(draw)
    save_image = SAVE_IMAGE(nrows = 1, ncols = 1, figTitle=title)
    save_image.addImage(cv2.cvtColor(_opencvImg, cv2.COLOR_BGR2RGB), title="")
    save_image.saveImage(RESULT_PATH+RESULT_DIR, "result")
    
    return


def GetBoundingBox(IMG_URL, CAM_RESULT, FEATURE_MAP, fig = [0, 0], dir_name = ""):
    
    
    global RESULT_PATH, RESULT_DIR, PIL_IMG, CV2_IMG
    RESULT_PATH = './Result/'
    RESULT_DIR = dir_name
    
    # check and make result dir to save result
    CheckDirExists(RESULT_PATH, RESULT_DIR)
    
    # load image
    PIL_IMG = Image.open(urllib2.urlopen(IMG_URL))
    CV2_IMG = cv2.cvtColor(np.array(PIL_IMG), cv2.COLOR_RGB2BGR)
    height, width, depth = CV2_IMG.shape
    
    # save base image
    save_image = SAVE_IMAGE(nrows = 1, ncols = 1, figTitle="base image")
    save_image.addImage(cv2.cvtColor(CV2_IMG, cv2.COLOR_BGR2RGB))
    save_image.saveImage(RESULT_PATH+RESULT_DIR, "base_image")
    
    # get CAM result bbox
    
    ## heatmap 얻기
    CAM_heatmaps = GetHeatmap(CAM_RESULT, height, width, 'CAM')
    CAM_heatmaps = GetGrayscaleHeatmap(CAM_heatmaps, 'CAM')
    
    ## contour와 bbox 비교 이미지 얻기
    CompareContourAndBBox(CAM_heatmaps, 'CAM')
    
    ## bbox 얻기
    CAM_BB = []
    for index, heatmap in enumerate(CAM_heatmaps):
        tmp_bb = GetBBox(heatmap)
        for index2, bbox in enumerate(tmp_bb):
            CAM_BB.append(bbox)
       
    title = "CAM_BBOX: "+str(len(CAM_BB))
    save_image = SAVE_IMAGE(nrows = 1, ncols = 1, figTitle=title)
    save_image.addImage(DrawBBox(CAM_BB, CV2_IMG))
    save_image.saveImage(RESULT_PATH+RESULT_DIR, "CAM_BBOX")
    
    
    
    # get FeatureMap bbox
    
    ## heatmap 얻기
    FM_heatmaps = GetHeatmap(FEATURE_MAP, height, width, 'FM', figSet = True, fig = fig)
    FM_heatmaps = GetGrayscaleHeatmap(FM_heatmaps, 'FM', figSet = True, fig = fig)
    
    ## contour와 bbox 비교 이미지 얻기
    CompareContourAndBBox(FM_heatmaps, 'FM', figSet = True, fig = fig)
    
    ## bbox 얻기
    FM_BB = []
    for index, heatmap in enumerate(FM_heatmaps):
        tmp_bb = GetBBox(heatmap)
        for index2, bbox in enumerate(tmp_bb):
            FM_BB.append(bbox)
      
    title = "FM_BBOX: "+str(len(FM_BB))
    save_image = SAVE_IMAGE(nrows = 1, ncols = 1, figTitle=title)
    save_image.addImage(DrawBBox(FM_BB, CV2_IMG))
    save_image.saveImage(RESULT_PATH+RESULT_DIR, "FM_BBOX")

    # get candidate bbox with CAM bbox and FeatureMap bbox
    candidate_bbox = GetCandidateBBox(FM_BB, CAM_BB)
    title = "candidate_bbox: "+str(len(candidate_bbox))
    save_image = SAVE_IMAGE(nrows = 1, ncols = 1, figTitle=title)
    save_image.addImage(DrawBBox(candidate_bbox, CV2_IMG))
    save_image.saveImage(RESULT_PATH+RESULT_DIR, "candidate_bbox")
    
    return candidate_bbox


def GetBoundingBox_SS(IMG_URL, CAM_RESULT, fig = [0, 0], dir_name = ""):
    
    
    global RESULT_PATH, RESULT_DIR, PIL_IMG, CV2_IMG
    RESULT_PATH = './Result/'
    RESULT_DIR = dir_name
    
    # check and make result dir to save result
    CheckDirExists(RESULT_PATH, RESULT_DIR)
    
    # load image
    PIL_IMG = Image.open(urllib2.urlopen(IMG_URL))
    CV2_IMG = cv2.cvtColor(np.array(PIL_IMG), cv2.COLOR_RGB2BGR)
    height, width, depth = CV2_IMG.shape
    
    # save base image
    save_image = SAVE_IMAGE(nrows = 1, ncols = 1, figTitle="base image")
    save_image.addImage(cv2.cvtColor(CV2_IMG, cv2.COLOR_BGR2RGB))
    save_image.saveImage(RESULT_PATH+RESULT_DIR, "base_image")
    
    # get CAM result bbox
    
    ## heatmap 얻기
    CAM_heatmaps = GetHeatmap(CAM_RESULT, height, width, 'CAM')
    CAM_heatmaps = GetGrayscaleHeatmap(CAM_heatmaps, 'CAM')
    
    ## contour와 bbox 비교 이미지 얻기
    CompareContourAndBBox(CAM_heatmaps, 'CAM')
    
    ## bbox 얻기
    CAM_BB = []
    for index, heatmap in enumerate(CAM_heatmaps):
        tmp_bb = GetBBox(heatmap)
        for index2, bbox in enumerate(tmp_bb):
            CAM_BB.append(bbox)
       
    title = "CAM_BBOX: "+str(len(CAM_BB))
    save_image = SAVE_IMAGE(nrows = 1, ncols = 1, figTitle=title)
    save_image.addImage(DrawBBox(CAM_BB, CV2_IMG))
    save_image.saveImage(RESULT_PATH+RESULT_DIR, "CAM_BBOX")
    
   
    # get SS bbox
    
    SS_BB = cv2_selective_search(CV2_IMG)
      
    title = "SS_BBOX: "+str(len(SS_BB))
    save_image = SAVE_IMAGE(nrows = 1, ncols = 1, figTitle=title)
    save_image.addImage(DrawBBox(SS_BB, CV2_IMG))
    save_image.saveImage(RESULT_PATH+RESULT_DIR, "SS_BBOX")

    # get candidate bbox with CAM bbox and FeatureMap bbox
    candidate_bbox = GetCandidateBBox(SS_BB, CAM_BB)
    title = "candidate_bbox: "+str(len(candidate_bbox))
    save_image = SAVE_IMAGE(nrows = 1, ncols = 1, figTitle=title)
    save_image.addImage(DrawBBox(candidate_bbox, CV2_IMG))
    save_image.saveImage(RESULT_PATH+RESULT_DIR, "candidate_bbox")
    
    return candidate_bbox
    
def R_CNN(IMG_URL, candidate_bbox, fig = [0, 0], dir_name = ""):
    
    global RESULT_PATH, RESULT_DIR, PIL_IMG, CV2_IMG
    RESULT_PATH = './Result/'
    RESULT_DIR = dir_name
    
    # check and make result dir to save result
    CheckDirExists(RESULT_PATH, RESULT_DIR)
    
    # load image
    PIL_IMG = Image.open(urllib2.urlopen(IMG_URL))
    CV2_IMG = cv2.cvtColor(np.array(PIL_IMG), cv2.COLOR_RGB2BGR)
    height, width, depth = CV2_IMG.shape
    # R-CNN
    ## load model
    global device, color
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    color = [[44, 195, 74], [62, 208, 80], [53, 230, 195], [20, 216, 183], [235, 220, 95], [16, 138, 103], [170, 172, 255], [17, 150, 98], [252, 125, 2], [142, 155, 193], [117, 25, 29], [235, 119, 120], [105, 211, 222], [66, 52, 154], [1, 33, 128], [72, 182, 183], [183, 35, 106], [216, 217, 0], [204, 201, 74], [39, 41, 236]]
    
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 20)

    model = model.to(device)
    model.load_state_dict(torch.load('./Model/Resnet50_ratio&size'))
    model.eval()
    
    det_probs = []

    for index, (x, y, w, h) in enumerate(candidate_bbox):
        area = (x, y, x + w, y + h)
        timage = PIL_IMG.crop(area)
        timage = data_transforms['test'](transforms.ToPILImage()(np.asarray(timage)))
        prob = get_predict(model, timage)
        det_probs.append(prob.tolist()[0])

    det_probs = torch.as_tensor(det_probs)
    
    final_probs = NMS(candidate_bbox, det_probs)
    DrawResult(candidate_bbox, final_probs)
    DrawResultByClass(candidate_bbox, final_probs)
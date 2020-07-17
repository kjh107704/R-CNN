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
PATH = BASE_PATH + "Images"
ANNOT = BASE_PATH + "Airplanes_Annotations"

# %%
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

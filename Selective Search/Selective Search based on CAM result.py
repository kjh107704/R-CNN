# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Parsing CAM data (.json)

# %%
import json 
import numpy as np
import cv2
from matplotlib import pyplot as plt


# %%
PATH = './sampleImg/'


# %%
with open(PATH+"liontiger.json") as json_file:
    json_data = json.load(json_file)

lion_data = np.array(json_data["lion"], dtype='uint8')
tiger_data = np.array(json_data["tiger"], dtype='uint8')

# %% [markdown]
# # heatmap 동작 확인

# %%
orig_heatmaps = []


img = cv2.imread(PATH+"lion_tiger.jpg")
height, width, _ = img.shape

# orig_heatmaps[0] = lion heatmap
orig_heatmaps.append(cv2.applyColorMap(cv2.resize(lion_data, (width, height)), cv2.COLORMAP_JET))
#result = orig_heatmaps[0] *0.7 + img + 0.5
#cv2.imshow(PATH+'lion_heatmap.jpg',result)

# orig_heatmaps[1] = tiger heatmap
orig_heatmaps.append(cv2.applyColorMap(cv2.resize(tiger_data, (width, height)), cv2.COLORMAP_JET))
#result = orig_heatmaps[1] *0.7 + img + 0.5
#cv2.imwrite(PATH+'tiger_heatmap.jpg',heatmap)

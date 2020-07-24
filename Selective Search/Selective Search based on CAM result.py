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

# %% [markdown]
# # color channel별 heatmap 뽑기

# %%
# orig_img에서 (R, G, B) 세 가지 채널의 정보 중 특정 채널의 정보만 남겨서 넘김
def get_channel_image(orig_img, channel):
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


# %%
# orig_heatmap에서 `R` 계열이 가장 중요한 feature 부분을 나타내므로 해당 정보만 사용
heatmap = getChannelImage(orig_heatmaps[1], 'r')
# orig_heatmap의 R 채널 데이터를 gray scale로 출력
plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY), cmap='gray')

# %% [markdown]
# # CAM 결과를 이용하여 Bounding Box 잡기

# %%
# color image를 gray scale로 바꾼 후, threshold를 적용함
# threshold는 고정 값으로 mean(min, max)
def get_grayscale_image_with_threshold(orig_img):
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    
    min_val = np.min(gray_img)
    max_val = np.max(gray_img)
    threshold = (min_val + max_val) / 2
    
    ret, gray_img = cv2.threshold(gray_img, threshold, 1, cv2.THRESH_BINARY)    
    
    return gray_img


# %%
# grayscale_mask 에서 1인 부분만 orig_img를 보여줌. 0인 부분은 검정색으로 보임
def get_masked_image(orig_img, gray_map):
    mask = cv2.cvtColor(gray_map, cv2.COLOR_GRAY2BGR)
    
    maskedRegion = np.where(mask == 1, img, 0)
    
    plt.imshow(cv2.cvtColor(maskedRegion, cv2.COLOR_BGR2RGB))


# %%
# gray scale img로 바꾸고, threshold 이상의 값만 binary로 살림
gray_map = getGrayscaleImageWithThreshold(heatmap)

showMaskedRegion(img, gray_map)


# %%
def get_contours(img_binary):
    
    return contours


# %%
def get_bounding_box(img_binary):
    bb = []
    
    contours = get_contours(img_binary)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bb.append([x, y, w, h])
        
    return bb


# %%
def draw_bounding_box(bounding_box, img):
    tmp_img = img.copy()
    for x, y, w, h in bounding_box:
        cv2.rectangle(tmp_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    plt.imshow(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))


# %%
def get_image_of_compare_contour_and_bounding_box(img_binary, img):
    
    tmp_img = img.copy()
    
    # draw contours - red
    for cnt in contours:
        cv2.drawContours(tmp_img, [cnt], 0, (0,0,255),3)
    
    # draw bounding box - green
    bb = getBoundingBox(img_binary)
    for x, y, w, h in bb:
        cv2.rectangle(tmp_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
    plt.imshow(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))


# %%
# contour 영역과 bounding box 비교
compareContourAndBoundingBox(gray_map, img)


# %%
bounding_box = getBoundingBox(gray_map)

drawBoundingBox(bounding_box, img)


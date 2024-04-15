import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
import torch
import torch.nn.functional as F
import cv2

path="F:/files/datas/2_Train"


ratios=[]

def preprocess_pic(img,size=(800,800)):
    img=torch.tensor(img).unsqueeze(0).unsqueeze(0)
    img=F.interpolate(img,size)
    return np.array(img[0,0])


def print_windows(img,img_dict):
    windows=img_dict['ann']
    for window in windows:
        y_1,x_1,y_2,x_2,_=window
        for i in range(int(x_1),int(x_2)):
            for j in range(5):
                img[i,int(y_1)+j]=255
                img[i,int(y_1)-j]=255
            for j in range(5):
                img[i, int(y_2) + j] = 255
                img[i, int(y_2) - j] = 255
        for i in range(int(y_1),int(y_2)):
            for j in range(5):
                img[int(x_1)+j, i] = 255
                img[int(x_1)-j, i] = 255
            for j in range(5):
                img[int(x_2) + j, i] = 255
                img[int(x_2) - j, i] = 255
    return img


img_s=[]
img_dics=[]

for i in range(1,6083):
    img=Image.open(os.path.join(path,f"pic_{i}.jpg"))
    img=np.array(img)
    if len(img.shape)==3:
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        plt.imshow(img)
        plt.show()

    ratios.append(np.array([800/img.shape[0],800/img.shape[1]]))
    img=preprocess_pic(img)
    f=open(os.path.join(path,f"pic_{i}.json"))
    img_dict=json.load(f)
    img_s.append(img)
    img_dics.append(img_dict)
img_s=np.array(img_s)
ratios=np.array(ratios)
print(ratios.shape)
np.save('train_pics.npy',img_s)
np.save('train_pic_ratios.npy',ratios)

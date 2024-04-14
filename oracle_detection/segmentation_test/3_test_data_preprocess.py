import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cv2

path='F:/files/datas/3_Test/Figures'



def preprocess_pic(img,size=(800,800)):
    img=torch.tensor(img).unsqueeze(0).unsqueeze(0)
    img=F.interpolate(img,size)
    return np.array(img[0,0])


train_pics=np.load('../Segmantation/train_pics.npy')
print(train_pics.shape)

img_s=[]
for root,_,file in os.walk(path):
    for filename in file:
        img=Image.open(os.path.join(root,filename))
        img=np.array(img)
        if len(img.shape) == 3:
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img=preprocess_pic(img)
        img_s.append(img)
img_s=np.array(img_s)
np.save('test_pics.npy',img_s)

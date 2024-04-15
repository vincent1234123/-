import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pic_preprocess.linear_gray_degree_transformation import RGB2gray,linear_gray_degree_transformation
from pic_preprocess.OTSU_byte_value import otsu_binarization
from Recognize.eight_connect_recognize import use_8_connect_to_clear_img


def img_reverse(img):
    H,W=img.shape
    if img[0:10,0:10].mean()<100 or img[H-1-10:H-1,0:10].mean()<100 or img[H-1-10:H-1,W-1-10:W-1].mean()<100 or img[0:10,W-1-10:W-1].mean()<100:
        return img
    height=img.shape[0]
    width=img.shape[1]
    for i in range(height):
        for j in range(width):
            gray = 255 - img[i, j]
            img[i, j] = np.uint8(gray)
    return img


def pic_preprocess(img,target_size=(224,224),mode='bilinear'):
    # plt.subplot(1,2,1)
    # plt.imshow(img,cmap='gray')
    ###RGB2gray
    img=RGB2gray(img)
    ###判断图像是否需要翻转
    img=img_reverse(img)
    ###otsu 2值化
    img=otsu_binarization(img)
    ###8 connect 筛选
    img=use_8_connect_to_clear_img(img)
    ###最后一步：插值
    img=torch.tensor(img)
    img=img.unsqueeze(0)
    img=img.unsqueeze(0)
    img=F.interpolate(img,size=target_size,mode=mode)
    img=np.array(img[0])
    # plt.subplot(1,2,2)
    # plt.imshow(img[0],cmap='gray')
    # plt.show()
    return img







import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

from linear_gray_degree_transformation import RGB2gray,linear_gray_degree_transformation
from OTSU_byte_value import otsu_binarization
from eight_connect import use_8_connect_to_clear_img

sys.setrecursionlimit(1000000) #例如这里设置为十万



def preprocess_img(img):
    img=RGB2gray(img)
    # img=linear_gray_degree_transformation(img)###选择使用
    img=otsu_binarization(img)
    poses,areas,labels,processed_img=use_8_connect_to_clear_img(img)
    return processed_img


for i in range(1,4):
    img=Image.open(f'F:/files/datas/1_Pre_test/pic_{i}.jpg')
    img=np.array(img)
    img_c=copy.deepcopy(img)
    processed_img=preprocess_img(img)
    plt.subplot(1,2,1)
    plt.imshow(img_c,cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(processed_img,cmap='gray')
    plt.show()

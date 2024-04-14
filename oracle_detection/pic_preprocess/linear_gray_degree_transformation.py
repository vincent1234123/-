import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2



def RGB2gray(img):
    if len(img.shape)==2:
        return img
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img




def linear_gray_degree_transformation(img):
    img=np.array(img)
    processed_img=np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (int(img[i, j] * 1.5) > 255):
                gray = 255
            else:
                gray = int(img[i, j] * 1.5)
            processed_img[i, j] = np.uint8(gray)
    return processed_img



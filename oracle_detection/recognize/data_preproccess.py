import os
from recognize_pic_preprocess import pic_preprocess
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt



path="data path"
def traversal_folder(folder_path):
    img_s=[]
    dirs=[]
    labels=[]
    kk=0
    id=-1
    for root,dirs_t,files in os.walk(folder_path):
        id+=1
        print(id)
        if len(dirs_t)!=0:
            dirs=dirs_t
        for file_name in files:
            if file_name.endswith('.bmp'):
                kk+=1
                print(kk)
                img=Image.open(os.path.join(root,file_name))
                img=np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img=pic_preprocess(img)###channel==1 h w
                img_s.append(img)
                labels.append(id)
    img_s=np.array(img_s)
    labels=np.array(labels)
    return img_s,labels


img_s,labels=traversal_folder(path)


ids=np.array([i for i in range(0,40617)])
np.random.shuffle(ids)
img_s=img_s[ids]
labels=labels[ids]

np.save('train_pics.npy',img_s[:37000])
np.save('train_labels.npy',labels[:37000])

np.save('test_pics.npy',img_s[37000:])
np.save('test_labels.npy',labels[37000:])

print(img_s.shape)
print(labels.shape)

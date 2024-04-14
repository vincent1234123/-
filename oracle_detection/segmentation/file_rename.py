import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from PIL import Image
import json



path="F:/files/datas/2_Train"
json_files=[]
img_s=[]


old_names=[]
new_names=[]

# for i in range(6217):
#     new_names.append(f"pic_{i+1}")



def traversal_folder(folder_path):
    id_pics = 0
    id_labels = 0
    for root,dirs_t,files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.jpg'):
                id_pics+=1
                img=Image.open(os.path.join(root,file_name))
                img=np.array(img)###channel 1
                img_s.append(img)
                old_names.append(file_name[:-4])




traversal_folder(path)
id=0
for old_name in old_names:
    if os.path.exists(os.path.join(path,old_name)+'.jpg') and os.path.exists(os.path.join(path, old_name) + '.json'):
        id+=1
        new_name=f"pic_{id}"
        os.rename(os.path.join(path,old_name)+'.jpg',os.path.join(path,new_name)+".jpg")
        os.rename(os.path.join(path, old_name) + '.json', os.path.join(path, new_name) + ".json")



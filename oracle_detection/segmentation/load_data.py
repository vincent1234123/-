import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
from PIL import Image

train_pics=np.load('train_pics.npy')
path="F:/files/datas/2_Train"
batch_size=64
ratios=np.load('train_pic_ratios.npy')



def load_data(batch_size=64):
    ids=np.random.choice(range(0,6082),batch_size)
    batch_pics=train_pics[ids]
    batch_pics=list(batch_pics)
    batch_labels=[]
    for id in ids:
        f=open(os.path.join(path,f'pic_{id+1}.json'))
        img_dict=json.load(f)
        img_dict['ann']=np.array(img_dict['ann'])
        ratio=ratios[id]
        img_dict['ann']=img_dict['ann']*np.array([ratio[1],ratio[0],ratio[1],ratio[0],1])
        img_dict['boxes']=torch.tensor(img_dict['ann'][:,:-1])
        img_dict['labels']=torch.tensor(img_dict['ann'][:,-1]).long()
        del img_dict['img_name']
        del img_dict['ann']
        batch_labels.append(img_dict)
    return batch_pics,batch_labels



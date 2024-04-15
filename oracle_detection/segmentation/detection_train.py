import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import os
import json
import matplotlib.pyplot as plt


device=torch.device('cuda')



###load data
train_pics=np.load('train_pics.npy')
path="Your data path"
batch_size=64
ratios=np.load('train_pic_ratios.npy')
def load_data(batch_size=64):
    ids=np.random.choice(range(0,5000),batch_size)
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


####load Net
num_classes=2 ##文字 背景
Net=fasterrcnn_resnet50_fpn(pretrained=True)
in_features = Net.roi_heads.box_predictor.cls_score.in_features
Net.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)


# state_dict=torch.load('model_detection.pt')
# Net.load_state_dict(state_dict)
Net=Net.to(device)






def model_train(model,lr,n_epochs):
    model.train()
    train_losses=[]
    optimizer=torch.optim.Adam(model.parameters(),lr)
    for i in range(n_epochs):
        X, Y = load_data(batch_size=8)
        X = [torch.tensor(xx).float().unsqueeze(0).to(device) for xx in X]
        Y = [{k: v.to(device) for k, v in t.items()} for t in Y]
        optimizer.zero_grad()
        loss_dict = model(X, Y)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        losses_value = losses.item()
        train_losses.append(losses_value)
        print(losses_value)
    plt.plot(train_losses)
    plt.show()

model_train(Net,0.0001,1000)
torch.save(Net.state_dict(),'model_detection.pt')





import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import os
import json
from draw import print_windows
import matplotlib.pyplot as plt
import copy

device=torch.device('cuda')

####load Net
num_classes=2 ##文字 背景
Net=fasterrcnn_resnet50_fpn(pretrained=True)
in_features = Net.roi_heads.box_predictor.cls_score.in_features
Net.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)


state_dict=torch.load('model_detection.pt')
Net.load_state_dict(state_dict)
Net=Net.to(device)



###load_data
train_pics=np.load('train_pics.npy')
path="F:/files/datas/2_Train"
batch_size=64
ratios=np.load('train_pic_ratios.npy')
def load_data(batch_size=64):
    ids=np.random.choice(range(5000,6082),batch_size)
    batch_pics=train_pics[ids]
    batch_pics=list(batch_pics)
    batch_labels=[]
    for id in ids:
        f=open(os.path.join(path,f'pic_{id+1}.json'))
        img_dict=json.load(f)
        img_dict['ann']=np.array(img_dict['ann'])
        ratio=ratios[id]
        img_dict['ann']=img_dict['ann']*np.array([ratio[1],ratio[0],ratio[1],ratio[0],1])
        # img_dict['boxes']=torch.tensor(img_dict['ann'][:,:-1])
        # img_dict['labels']=torch.tensor(img_dict['ann'][:,-1]).long()
        # del img_dict['img_name']
        # del img_dict['ann']
        batch_labels.append(img_dict)
    return batch_pics,batch_labels


def NMS(boxes,scores,tolerance_rate=0.5):
    def compute_coverage(box_1,box_2):
        y_1, x_1,y_2,x_2=box_1
        y_3,x_3,y_4,x_4=box_2
        area_1=(x_2-x_1)*(y_2-y_1)
        area_2=(x_4-x_3)*(y_4-y_3)
        d_x=max(min(x_4,x_2)-max(x_1,x_3),0)
        d_y=max(min(y_4,y_2)-max(y_1,y_3),0)
        coverage_area=d_x*d_y
        return (coverage_area/area_1+coverage_area/area_2)/2

    ###prefer small frame
    label=np.ones_like(scores).astype(np.int32)
    num_boxes=boxes.shape[0]
    ids=[]
    for i in range(num_boxes):
        if label[i]==0:
            continue
        for j in range(num_boxes):
            if label[j]==0:
                continue
            for k in range(num_boxes):
                if label[k]==0:
                    continue
                if compute_coverage(boxes[i],boxes[j])>tolerance_rate and compute_coverage(boxes[i],boxes[k])>tolerance_rate and compute_coverage(boxes[j],boxes[k])<0.2:
                    label[i]=0
    for i in range(num_boxes):
        if label[i]==1:
            ids.append(i)
    boxes=boxes[ids]
    scores=scores[ids]
    ###NMS
    ids=np.ones_like(scores).astype(np.int32)
    nms_boxes=None
    nms_scores=None
    sorted_ids=sorted(range(len(scores)),reverse=True,key=lambda k:scores[k])
    boxes=boxes[sorted_ids]
    scores=scores[sorted_ids]
    num_boxes=boxes.shape[0]
    for i in range(num_boxes):
        if ids[i]==1:
            for j in range(i+1,num_boxes):
                if ids[j]==1 and compute_coverage(boxes[i],boxes[j])>tolerance_rate:
                    ids[j]=0
    return_ids=[]
    for i in range(num_boxes):
        if ids[i]==1:
            return_ids.append(i)
    boxes=boxes[return_ids]
    scores=scores[return_ids]
    return boxes,scores


X,Y=load_data()
Net.eval()
for img,label in zip(X,Y):
    img_c=copy.deepcopy(img)
    plt.subplot(1,3,1)
    plt.imshow(img,cmap='gray')

    plt.subplot(1,3,2)
    img_truth=print_windows(img,label['ann'])
    plt.imshow(img_truth,cmap='gray')

    img=torch.tensor(img).float().to(device).unsqueeze(0).unsqueeze(0)
    res=Net(img)
    res=res[0]
    boxes=res['boxes'].cpu().detach().numpy()
    labels=res['labels'].cpu().detach().numpy()
    scores=res['scores'].cpu().detach().numpy()
    ids=[]
    for i in range(scores.shape[0]):
        if scores[i]>0.8 and labels[i]==1:
            ids.append(i)
    boxes=boxes[ids]
    labels=labels[ids]
    scores=scores[ids]
    boxes,scores=NMS(boxes,scores)

    plt.subplot(1,3,3)
    img_predict=print_windows(img_c,boxes)
    plt.imshow(img_predict,cmap='gray')


    plt.show()


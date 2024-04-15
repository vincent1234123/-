import numpy as np
import torch
from PIL import Image
import os
import torch.nn.functional as F
from Segmantation.draw import print_windows
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import copy
import matplotlib.pyplot as plt
from last_question_data_preprocess import pic_preprocess




device=torch.device('cuda')

num_classes=2
Net=fasterrcnn_resnet50_fpn(pretrained=True)
in_features=Net.roi_heads.box_predictor.cls_score.in_features
Net.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)


state_dict=torch.load('Segmantation/model_detection.pt')
Net.load_state_dict(state_dict)
Net=Net.to(device)

test_pics=np.load('last_question_pics.npy')
print(test_pics.shape)



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

    ###prefer small frame(模型的局限性)
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

X=list(test_pics)
Net.eval()
pic_letter_nums = []
pic_img_s=[]
id=0
for img in X:
    id+=1
    img_pics=[]
    img_c=copy.deepcopy(img)
    plt.subplot(1,2,1)
    plt.imshow(img,cmap='gray')

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
    boxes, scores = NMS(boxes, scores)#####y1 x1 y2 x2
    num_boxes=boxes.shape[0]
    pic_letter_nums.append(num_boxes)

    for i in range(num_boxes):
        box=boxes[i]
        # plt.subplot(1, 2, 1)
        # plt.imshow(img_c,cmap='gray')
        # plt.subplot(1, 2, 2)
        img_kk=copy.deepcopy(img_c[int(box[1]):int(box[3]),int(box[0]):int(box[2])])
        # plt.imshow(img_kk,cmap='gray')
        # plt.show()
        img_kk=pic_preprocess(img_kk)
        img_pics.append(img_kk)
    img_pics=np.array(img_pics)
    np.save(f'last_question_datas/pic_{id}.npy',img_pics)
    print(img_pics.shape)
    pic_img_s.append(img_pics)



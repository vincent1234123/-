import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from load_data import load_data
device=torch.device('cuda')







# def draw_frames(img,img_dict,line_width=3):
#     red=torch.tensor([255,0,0])
#     windows=img_dict['boxes']
#     img=img[0]
#     img=img.permute(1,2,0)
#     windows=windows.cpu().detach()
#     h=img.shape[0]
#     w=img.shape[1]
#     for i in range(windows.shape[0]):
#         y_1,x_1,y_2,x_2=windows[i]
#         for i in range(int(x_1),int(x_2)):
#             for j in range(1,line_width+1):
#                 if int(y_1)+j>=w:
#                     break
#                 img[i,int(y_1)+j]=red
#
#             for j in range(1,line_width+1):
#                 if int(y_2)-j<0:
#                     break
#                 img[i, int(y_2) - j] = red
#
#         for i in range(int(y_1),int(y_2)):
#             for j in range(1,line_width+1):
#                 if int(x_1)+j>=h:
#                     break
#                 img[int(x_1)+j, i] = red
#
#             for j in range(1,line_width+1):
#                 if int(x_2)-j<0:
#                     break
#                 img[int(x_2)- j, i] = red
#
#     img=img.permute(2,0,1)
#     return img




num_classes=2###text or background

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)
model.train()
model=model.to(device)



X,Y=load_data(batch_size=2)
X=[torch.tensor(xx).float().unsqueeze(0).to(device) for xx in X]
Y=[{k: v.to(device) for k, v in t.items()} for t in Y]

res=model(X,Y)
print(res)

torch.save(model.state_dict(),'model_detection.pt')


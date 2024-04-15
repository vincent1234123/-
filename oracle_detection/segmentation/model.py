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


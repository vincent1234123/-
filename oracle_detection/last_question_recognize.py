import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn.functional as F
from Recognize.resnet50 import resnet50
device=torch.device('cuda')


f=open('duizhaobiao.txt')
text=f.readline()

matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号

test_original_pics=np.load('last_question_pics.npy')
test_detection_pics=[]
for i in range(1,51):
    detection_pic=np.load(f'last_question_datas/pic_{i}.npy')
    test_detection_pics.append(detection_pic)


# ###draw
# for i in range(50):
#     original_pic=test_original_pics[i]
#     detection_pic=test_detection_pics[i]
#     for j in range(detection_pic.shape[0]):
#         pic=detection_pic[j][0]
#         plt.subplot(1,2,1)
#         plt.imshow(original_pic,cmap='gray')
#         plt.subplot(1,2,2)
#         plt.imshow(pic,cmap='gray')
#         plt.show()

Net=resnet50(1)
state_dict=torch.load('Recognize/model.pt')
Net.load_state_dict(state_dict)
Net=Net.to(device)

labels=[]
ppp=0
for original_id,detection_pic in enumerate(test_detection_pics):
    ppp+=1
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.imshow(test_original_pics[original_id],cmap='gray')
    plt.axis('off')  # 去掉坐标轴

    detection_pic=torch.tensor(detection_pic).float().to(device)
    res=Net(detection_pic)
    # res=res.cpu().detach().numpy()

    #####top_1
    res_top_1=torch.argmax(res,dim=1)

    ######top_5
    res_top_5=torch.topk(res,k=5,dim=1,largest=True)[1]

    ###
    ans_top_1=[]
    ans_top_5=[]
    res_top_1=res_top_1.cpu().detach().numpy()
    res_top_5=res_top_5.cpu().detach().numpy()
    for id in list(res_top_1):
        ans_top_1.append(text[id])



    for i in range(res_top_5.shape[0]):
        text_t=""
        for id in list(res_top_5[i]):
            text_t+=text[id]
        ans_top_5.append(text_t)

    for id,pic in enumerate(detection_pic):
        # print(ans_top_1[id])
        # print(ans_top_5[id])
        if id>=10:
            break
        pic=pic.cpu().detach().numpy()
        plt.subplot(4,5,11+id)
        plt.title(ans_top_5[id],fontsize='x-small')
        plt.imshow(pic[0],cmap='gray')
        plt.axis('off')  # 去掉坐标轴
    print(ppp)
    plt.savefig(f'figures/fig_{ppp}.jpg')


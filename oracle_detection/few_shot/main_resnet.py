import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import DN4
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torchvision
from tqdm import tqdm

device=torch.device('cuda')

names=[]
train_pics=np.load('../Recognize/train_pics.npy')
test_pics=np.load('../Recognize/test_pics.npy')
pics=np.concatenate((train_pics,test_pics),axis=0)
train_labels=np.load('../Recognize/train_labels.npy')
test_labels=np.load('../Recognize/test_labels.npy')
labels=np.hstack([train_labels,test_labels])
print(pics.shape)
print(labels.shape)
ids=sorted(range(len(labels)),key=lambda k:labels[k])
labels=labels[ids]
pics=pics[ids]

img_categories=[]
file_nums=[]
def traversal_folder(folder_path):
    img_s=[]
    files_sum=0
    for root,dirs,files in os.walk(folder_path):
        num_files=len(files)
        if num_files==0:
            continue
        files_sum+=num_files
        file_nums.append(num_files)
    return img_s
traversal_folder("F:/files/datas/4_Recognize/train")
la=0

for file_num in file_nums:
    img_categories.append(pics[la:la+file_num])
    la=la+file_num



n_class_train=76
n_class_test=76


img_categories=[torch.tensor(cat).float() for cat in img_categories]

# for i in range(100):
#     plt.imshow(pics[i][0],cmap='gray')
#     plt.show()
#query:20 support_set 5 1
def load_data(img_s,n_classes):#five way one shot
    ##support set 5 category
    support_set_id=np.random.choice(range(0,n_classes),5)
    data_imgs=[]
    for i in range(5):
        data_imgs.append(np.random.choice(range(0,len(img_categories[support_set_id[i]])),6))
    support_set=[]
    query_set=[]
    for i in range(5):
        data_img=data_imgs[i]
        for j in data_img[:5]:
            query_set.append(img_s[support_set_id[i]][j].unsqueeze(0))
        support_set.append(img_s[support_set_id[i]][data_img[5]].unsqueeze(0).unsqueeze(0))
    labels=[]
    for i in range(25):
        labels.append(i//5)
    labels=np.array(labels)

    id=np.arange(0,25)
    np.random.shuffle(id)
    support_set=torch.cat(support_set,dim=0)
    query_set=torch.cat(query_set,dim=0)
    labels=torch.tensor(labels)

    support_set=support_set.squeeze(1)
    # for i in range(5):
    #     for j in range(5):
    #         k=i*5+j
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(support_set[i][0])
    #         plt.subplot(1,2,2)
    #         plt.imshow(query_set[k][0])
    #         plt.show()
    query_set=query_set[id]
    labels=labels[id]
    query_set=query_set[:10]
    labels=labels[:10]
    for kk in range(10):
        for i in range(support_set.shape[0]):
            pic_s=support_set[i][0]
            plt.subplot(2,5,6+i)
            plt.imshow(pic_s,cmap='gray')
            plt.axis("off")
            plt.title(f"support set {i+1}",fontsize='x-small')
        plt.subplot(2,1,1)
        plt.imshow(query_set[kk][0],cmap='gray')
        plt.axis("off")
        plt.title("ans: "+str(int((labels[kk]+1).detach().numpy())))
        plt.show()
    return query_set,support_set,labels



q,s,labels=load_data(img_categories,76)
print(q.shape)
print(s.shape)
print(labels.shape)
# print(q)
# print(s)
# print(labels)
def model_predict(model,n=300):
    model.eval()
    num_sum=0
    num_correct=0
    for i in range(n):
        data=load_data(img_categories,76)
        X_1 = data[0].to(device)
        X_2 = data[1].to(device)
        X_2=X_2.unsqueeze(1)
        Y = data[2].to(device)
        res=model(X_1,X_2)
        res=torch.argmax(res,dim=1)
        num_sum+=X_1.shape[0]
        for p,t in zip(res,Y):
            if p==t:
                num_correct+=1
    accuracy=num_correct/num_sum*100
    print("========================================")
    print("sum:",num_sum)
    print("correct:",num_correct)
    print("accuracy:",accuracy)
    print("=========================================")
    return accuracy



def train_model(model,lr=0.001,iterations=10,loss_fn=nn.CrossEntropyLoss()):
    model.train()
    optimizer=optim.SGD(model.parameters(),lr=lr)
    losses=[]
    accuracies=[]
    for num_iter in range(iterations):
        model.train()
        data=load_data(img_categories,76)
        X_1=data[0].to(device)
        X_2=data[1].to(device)
        Y=data[2].to(device)
        X_2=X_2.unsqueeze(1)
        optimizer.zero_grad()
        res=model(X_1,X_2)
        loss=loss_fn(res,Y.long())
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
        print(f"epoch {num_iter+1}:",losses[-1])
        if num_iter%20==0:
            accuracy=model_predict(model,n=5)
            accuracies.append(accuracy)
    return losses,accuracies


net = DN4.define_DN4Net(which_model='ResNet256F')

for m in net.modules():
    if isinstance(m,(nn.Conv2d,nn.Linear)):
        nn.init.xavier_uniform_(m.weight)

state_dict=torch.load('model_resnet.pt')
net.load_state_dict(state_dict)

net=net.to(device)
losses,accuracies=train_model(net,lr=0.0001,iterations=100)
plt.subplot(1,2,1)
plt.plot(losses)

plt.subplot(1,2,2)
plt.plot(accuracies)
plt.show()
net.eval()
score=model_predict(net)
print('last accuracy:',score)


# torch.save(net.state_dict(), 'model_resnet.pt')



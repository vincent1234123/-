import numpy as np
import matplotlib.pyplot as plt
import torch

train_pics=np.load('train_pics.npy')
train_labels=np.load('train_labels.npy')
print(train_pics.shape)
print(train_labels.shape)
print(train_labels)

for i in range(1000):
    pic=train_pics[i]
    pic=pic[0]
    print(train_labels[i])
    plt.imshow(pic,cmap='gray')
    plt.show()
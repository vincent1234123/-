import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


import sys
sys.setrecursionlimit(1000000)


def use_8_connect_to_clear_img(img,p_max=10,p_min=0.1):
    area_t = 0
    min_x=10000
    max_x=-1000
    min_y=10000
    max_y=-10000
    def dfs(x,y,id):
        nonlocal area_t
        nonlocal min_x
        nonlocal min_y
        nonlocal max_x
        nonlocal max_y
        if x<0 or y<0 or x>=H or y>=W:
            return
        if labels[x,y]!=0:
            return
        if img[x,y]!=1:
            return
        labels[x,y]=id
        area_t+=1
        min_x=min(min_x,x)
        max_x=max(max_x,x)
        min_y=min(min_y,y)
        max_y=max(max_y,y)
        dfs(x+1,y+1,id)
        dfs(x+1,y,id)
        dfs(x+1,y-1,id)
        dfs(x,y+1,id)
        dfs(x,y-1,id)
        dfs(x-1,y+1,id)
        dfs(x-1,y,id)
        dfs(x-1,y-1,id)

    block_poses=[]
    block_areas=[]
    block_frames=[]
    H,W=img.shape
    labels=np.zeros_like(img).astype(np.int32)###保存有没有遍历到
    id=0
    for i in range(H):
        for j in range(W):
            if labels[i,j]==0 and img[i,j]==1:
                id+=1
                dfs(i,j,id)
                block_poses.append(np.array([i,j]))
                block_areas.append(area_t)
                block_frames.append(np.array([min_x,min_y,max_x,max_y]))
                area_t=0
                min_x = 10000
                max_x = -1000
                min_y = 10000
                max_y = -10000
    block_areas=np.array(block_areas)
    area_mean=block_areas.mean()
    area_std=block_areas.std()
    paint_labels=np.zeros_like(img).astype(np.int8)
    def dfs_paint(x,y):
        if x<0 or x>=H:
            return
        if y<0 or y>=W:
            return
        if paint_labels[x,y]==1:
            return
        if img[x,y]==0:
            return
        img[x,y]=0
        paint_labels[x,y]=1
        dfs_paint(x+1,y)
        dfs_paint(x+1,y+1)
        dfs_paint(x+1,y-1)
        dfs_paint(x,y+1)
        dfs_paint(x,y-1)
        dfs_paint(x-1,y)
        dfs_paint(x-1,y+1)
        dfs_paint(x-1,y-1)
    for id,pos in enumerate(block_poses):
        block_area=block_areas[id]
        block_frame=block_frames[id]
        d_x=block_frame[1]-block_frame[0]+1
        d_y=block_frame[3]-block_frame[2]+1
        if block_area<20:
            dfs_paint(pos[0],pos[1])


    return img





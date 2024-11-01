from datetime import datetime
import os
import shutil 
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import  random
from torch.utils.data import DataLoader, random_split

#得到数据集的编号
def get_file_num(root_path): 
    files_num=[]
    for dir_name in os.listdir(root_path):
        #获取目录或文件的路径
        file_path = os.path.join(root_path,dir_name)       
        pathsplit = os.path.splitext(file_path)
        path_list=pathsplit[0].split('/')
        files_num.append(path_list[-1])
    return files_num

#将文件名列表按指定比例percent随机划分为训练集图片名列表和验证集图片名列表
def split_dateset_name(file_names,percent):
    n_val = int(len(file_names) * percent)
    n_train = len(file_names) - n_val
    train_name,val_name=random_split(file_names, [n_train,n_val])
    return train_name,val_name
    # print(train_name[5],len(train_name))

#将文件srcfile移动到指定文件夹dstpath
def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + fname))
 

data_path='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/train'
image='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/train/image/'

#目的路径记得最后加反斜杠！！！
# new_train_path='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/date_8_2/train/image/'
# new_train_label_path='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/date_8_2/train/label/'
# new_val_path='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/date_8_2/val/image/'
# new_val_label_path='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/date_8_2/val/label/'

new_train_path='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/date_9_1_3/train/image/'
new_train_label_path='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/date_9_1_3/train/label/'
new_val_path='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/date_9_1_3/val/image/'
new_val_label_path='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/date_9_1_3/val/label/'

file_names=get_file_num(image)#获取数据集图片名字
train_name,val_name=split_dateset_name(file_names,0.1)#划分训练集和验证集名字
# print(len(train_name))

for i in range (len(train_name)):
    image_path=data_path+'/image/'+train_name[i]+'.jpg'
    label_path=data_path+'/label/'+train_name[i]+'_segmentation.png'
    mycopyfile(image_path, new_train_path)
    mycopyfile(label_path, new_train_label_path)

for i in range (len(val_name)):
    image_path=data_path+'/image/'+val_name[i]+'.jpg'
    label_path=data_path+'/label/'+val_name[i]+'_segmentation.png'
    mycopyfile(image_path, new_val_path)
    mycopyfile(label_path, new_val_label_path)







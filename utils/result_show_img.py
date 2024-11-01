from datetime import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2

def union_image_mask(image_path, model_path,pre_path,label_path, num):
    # 读取原图
    image = cv2.imread(image_path)
    print(image.shape) # (400, 500, 3)
    # print(image.size) # 600000
    # print(image.dtype) # uint8

    # 读取分割mask，这里本数据集中是白色背景黑色mask
    image_pre = cv2.imread(pre_path, cv2.IMREAD_GRAYSCALE)
    print(image_pre.shape) 
    # 裁剪到和原图一样大小
    image_pre = image_pre[0:767, 0:1022]
    h, w = image_pre.shape
    # cv2.imshow("2d", mask_2d)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ret_pre, thresh_pre = cv2.threshold(image_pre, 127, 255, 0)
    contours_pre, hierarchy_pre = cv2.findContours(thresh_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    image_label = image_label[0:767, 0:1022]
    hl, wl = image_label.shape
    ret_label, thresh_label = cv2.threshold(image_label, 127, 255, 0)
    contours_label, hierarchy_label = cv2.findContours(thresh_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # draw_img_label=cv2.drawContours(image, contours_label, -1, (238, 204, 0), 1)
    draw_img_label=cv2.drawContours(image, contours_label, -1, (119, 255, 0), 2)
    draw_img_pre=cv2.drawContours(image, contours_pre, -1,(60, 20,220 ) , 2)

    # draw_img_pre=cv2.drawContours(image, contours_pre, -1,(0, 0, 255) , 1)
    # draw_img_label=cv2.drawContours(image, contours_label, -1, (0, 255, 0), 1)

    # # 打开画了轮廓之后的图像
    # cv2.imshow('pre', draw_img_pre)
    # # cv2.imshow('label', draw_img_label)
    # k = cv2.waitKey(0)
    # if k == 27:
    #     cv2.destroyAllWindows()
    # 保存图像
    save_path='/home/fzz/huaner/codes/Pytorch-UNet-master/modules/save_study1/unet_se_gn/'
    # model_path=save_path
    # cv2.imwrite(model_path+'batch'+str(num)+'_result'+ ".bmp", image)
    cv2.imwrite(save_path+'batch'+str(num)+'_result'+ ".bmp", image)
    print('finish'+str(num))

def draw_line(pre_path,image,color):
    image_pre = cv2.imread(pre_path, cv2.IMREAD_GRAYSCALE)
    image_pre = image_pre[0:767, 0:1022]
    h, w = image_pre.shape
    ret_pre, thresh_pre = cv2.threshold(image_pre, 127, 255, 0)
    contours_pre, hierarchy_pre = cv2.findContours(thresh_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    draw_img_unet_bn=cv2.drawContours(image, contours_pre, -1,color, 1)

def union_image_mask_all(image_path, model_path,pre_path,pre_path_list,label_path, num,save_path):
    # 读取原图
    image = cv2.imread(image_path)

    image_pre = cv2.imread(pre_path, cv2.IMREAD_GRAYSCALE)
    image_pre = image_pre[0:767, 0:1022]
    h, w = image_pre.shape
    ret_pre, thresh_pre = cv2.threshold(image_pre, 127, 255, 0)
    contours_pre, hierarchy_pre = cv2.findContours(thresh_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    image_label = image_label[0:767, 0:1022]
    hl, wl = image_label.shape
    ret_label, thresh_label = cv2.threshold(image_label, 127, 255, 0)
    contours_label, hierarchy_label = cv2.findContours(thresh_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    draw_line(pre_path_list[0],image,(28, 28,28 ))
    draw_line(pre_path_list[1],image,(0,0,205 ) )
    # draw_line(pre_path_list[2],image,(255,127,0 ) )
    draw_line(pre_path_list[3],image,(255,165,0 ) )
    # draw_line(pre_path_list[4],image,(255,255,0 ) )
    draw_line(pre_path_list[5],image,(0, 255,255 ))
    draw_img_label=cv2.drawContours(image, contours_label, -1, (119, 255, 0), 2)
    draw_img_pre=cv2.drawContours(image, contours_pre, -1,(60, 20,220 ) , 2)

    # draw_img_pre=cv2.drawContours(image, contours_pre, -1,(0, 0, 255) , 1)
    # draw_img_label=cv2.drawContours(image, contours_label, -1, (0, 255, 0), 1)

    # 打开画了轮廓之后的图像
    cv2.imshow('pre', draw_img_pre)
    cv2.imshow('label', draw_img_label)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()

    # 保存图像
    cv2.imwrite(save_path+'batch'+str(num)+'_result'+ ".bmp", image)
    print('finish'+str(num))


proposed_path='/home/fzz/huaner/codes/Pytorch-UNet-master/modules/save_modules/20220310_191834_cbam_aspp_isic2018_SZ_256_LR_0.0001_BS_4_WD1e-08_GN_64_RT_[6, 12, 18]_Pool_False/epoch29/test11/'

unet_bn='/home/fzz/huaner/codes/Pytorch-UNet-master/modules/save_modules/20220311_195622_unet_isic2018_SZ_256_LR_0.0001_BS_4_WD1e-08_GN_64_RT_[6, 12, 18]_Pool_False/epoch44/test11/'
unet_gn='/home/fzz/huaner/codes/Pytorch-UNet-master/modules/save_modules/20220320_153123_unet_isic2018_SZ_256_LR_0.0001_BS_4_WD1e-08_GN_64_RT_[6, 12, 18]_Pool_False/epoch28/test10/'
unet_aspp_bn='/home/fzz/huaner/codes/Pytorch-UNet-master/modules/save_modules/20220314_192619_aspp_unet_isic2018_SZ_256_LR_0.0001_BS_4_WD1e-08_GN_64_RT_[6, 12, 18]_Pool_False/epoch50/test10/'
unet_aspp_gn='/home/fzz/huaner/codes/Pytorch-UNet-master/modules/save_modules/20220320_202204_aspp_unet_isic2018_SZ_256_LR_0.0001_BS_4_WD1e-08_GN_64_RT_[6, 12, 18]_Pool_False/epoch96/test10/'
unet_cbam_bn='/home/fzz/huaner/codes/Pytorch-UNet-master/modules/save_modules/20220311_213806_cbam_isic2018_SZ_256_LR_0.0001_BS_4_WD1e-08_GN_64_RT_[6, 12, 18]_Pool_False/epoch25/test10/'
unet_cbam_gn='/home/fzz/huaner/codes/Pytorch-UNet-master/modules/save_modules/20220317_213404_cbam_isic2018_SZ_256_LR_0.0001_BS_4_WD1e-08_GN_64_RT_[6, 12, 18]_Pool_False/epoch40/test10/'

save_path='/home/fzz/huaner/codes/Pytorch-UNet-master/modules/save_study1/all/'

# model_path=proposed_path
model_path='/home/fzz/huaner/codes/Pytorch-UNet-master/modules/save_modules/20220530_091822_se_unet_isic2018_SZ_256_LR_0.0001_BS_4_WD1e-08_GN_64_RT_[6, 12, 18]_Pool_False/epoch22/test10/'

for i in range (64):
    num=i+1
    image_path=model_path+'batch'+str(num)+'_img.png'
    pre_path=model_path+'batch'+str(num)+'_pred.png'

    label_path=model_path+'batch'+str(num)+'_mask.png'
    union_image_mask(image_path, model_path,pre_path, label_path,i+1)

    # pre_list=[]
    # pre_path1=unet_bn+'batch'+str(num)+'_pred.png'
    # pre_path2=unet_gn+'batch'+str(num)+'_pred.png'
    # pre_path3=unet_aspp_bn+'batch'+str(num)+'_pred.png'
    # pre_path4=unet_aspp_gn+'batch'+str(num)+'_pred.png'
    # pre_path5=unet_cbam_bn+'batch'+str(num)+'_pred.png'
    # pre_path6=unet_cbam_gn+'batch'+str(num)+'_pred.png'
    # pre_list.append(pre_path1)
    # pre_list.append(pre_path2)
    # pre_list.append(pre_path3)
    # pre_list.append(pre_path4)
    # pre_list.append(pre_path5)
    # pre_list.append(pre_path6)
    # union_image_mask_all(image_path, model_path,pre_path,pre_list, label_path,i+1,save_path)
    


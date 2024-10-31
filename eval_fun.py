from os import PRIO_PGRP
import torch
from torch.autograd import Function

#参考：https://blog.csdn.net/qq_36201400/article/details/109180060
    # TN=((~output_) &(~ target_)).sum() 
    # FP=((output_) &( ~target_)).sum() 
    # TP=intersection
    # FN=target_.sum()-intersection
    # TNFP=(~target_).sum()
    # TPFN=target_.sum()

#1.1Dice系数   
def dice_coef(output, target):#output为预测结果 target为真实结果
    smooth = 1e-5 #防止0除
 
    if torch.is_tensor(output):
        # output = torch.sigmoid(output).data.cpu().numpy()  
        # sigmoid激活后与原代码结果不一致,原因是在eval_module文件中已经进行了sigmoid
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    # print("output:{}".format(output))
    # print(output.shape)

    # print("target:{}".format(target))
    # print(target.shape)
    
    # #法1：*实现按位相乘
    # # intersection= (output * target).sum()
    # # return (2. * intersection + smooth) / \
    # #     (output.sum() + target.sum() + smooth)

    # 法2：&实现按位与运算
    output_ = output > 0.5
    target_ = target > 0.5 
    # print("output_:{}".format(output_))
    # print(output_.shape)
    # print("target_:{}".format(target_))
    # print(target_.shape)

    intersection = (output_ & target_).sum()
    return (2. * intersection + smooth) / \
        (output_.sum() + target_.sum() + smooth)


#1.2Jassard Index
def jassard(output, target):
    smooth = 1e-5 #防止0除
 
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    output_ = output > 0.5
    target_ = target > 0.5 

    intersection = (output_ & target_).sum()
    return (intersection + smooth) / \
        (output_.sum() + target_.sum()-intersection + smooth)

#1.3 VOE体积重叠误差
def voe(output, target):
    smooth = 1e-5 #防止0除
 
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    #intersection = (output * target).sum()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()

    v=(intersection + smooth) / \
        (output_.sum() + target_.sum() -intersection+ smooth)
    return  1-v

#1.4 RVD相对体积差异
def rvd(output, target):
    smooth = 1e-5 #防止0除
 
 
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    return (output.sum() - target.sum()+ smooth) / \
       (target.sum() + smooth)

#1.5 ASD平均对称面距离

#1.6 MSD最大对称表面距离


###################################################################################

#2.IoU重叠度/交并比=Jassard?????
def iou_score(output, target):
    smooth = 1e-5
 
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()  #交集
    union = (output_ | target_).sum()  #并集
    
    TN=((~output_) &(~ target_)).sum() 
    FP=((output_) &( ~target_)).sum() 
    TP=intersection
    FN=target_.sum()-intersection
    TNFP=(~target_).sum()
    TPFN=target_.sum()

    return (TP + smooth) / \
        (FP+TP+FN+ smooth)

# def iou_score(output, target):
#     smooth = 1e-5
 
#     if torch.is_tensor(output):
#         #output = torch.sigmoid(output).data.cpu().numpy()
#         output = output.data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     output_ = output > 0.5
#     target_ = target > 0.5
#     intersection = (output_ & target_).sum()  #交集
#     union = (output_ | target_).sum()  #并集
 
#     return (intersection + smooth) / (union + smooth)


#########################################################################################

#3.1 Sensitivity灵敏度/Recall召回率/TPR真阳性率/////////////////////
def sensitivity(output, target):
    smooth = 1e-5
 
    if torch.is_tensor(output):
        #output = torch.sigmoid(output).data.cpu().numpy()
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
 
    # intersection = (output * target).sum()

    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()  #交集
 
    return (intersection + smooth) / \
        (target_.sum() + smooth)

#3.2 /4.1Specificity特异度/////////////////////////////////////////////
def specificity(output, target):
    smooth = 1e-5
 
    if torch.is_tensor(output):
        #output = torch.sigmoid(output).data.cpu().numpy()
        output = output.data.cpu().numpy()

    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()  #交集
    union = (output_ | target_).sum()  #并集
    
    TN=((~output_) &(~ target_)).sum() 
    FP=((output_) &( ~target_)).sum() 
    TP=intersection
    FN=target_.sum()-intersection
    TNFP=(~target_).sum()
    TPFN=target_.sum()
    # print("TN+FP:{};TNFP:{}".format(TN+FP,TNFP))
    # print("sprcifity:tp:{};tn:{};fp:{};fn:{}".format(TP,TN,FP,FN))
    return (TN + smooth) / \
        (TNFP+ smooth)

#4.2 Precision精确率/查准率///////////////////////////////////////////
def precision(output, target):
    smooth = 1e-5
 
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
 
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()  #交集
    union = (output_ | target_).sum()  #并集
    
    TN=((~output_) &(~ target_)).sum() 
    FP=((output_) &( ~target_)).sum() 
    TP=intersection
    FN=target_.sum()-intersection
    TNFP=(~target_).sum()
    TPFN=target_.sum()

    return (TP+smooth) / \
        (FP+TP+smooth)

#4.3 Recall召回率/查重率/////////////////////////////////////////////
def recall(output, target):
    smooth = 1e-5
 
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
 
    # intersection = (output * target).sum()
    # return (intersection + smooth) / \
    #     (output.sum() + smooth)

    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()  #交集
    union = (output_ | target_).sum()  #并集

    TN=((~output_) &(~ target_)).sum() 
    FP=((output_) &( ~target_)).sum() 
    TP=intersection
    FN=target_.sum()-intersection
    TNFP=(~target_).sum()
    TPFN=target_.sum()
    # print("Recall:tp:{};tn:{};fp:{};fn:{}".format(TP,TN,FP,FN))
    return (TP + smooth) / \
        (TP+FN+ smooth)


#4.4 Accuracy准确率///////////////////////////////////////////////////
def accuracy(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        # output = torch.sigmoid(output).data.cpu().numpy()  
        # sigmoid激活后与原代码结果不一致,原因是在eval_module文件中已经进行了sigmoid
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    # print("output:{}".format(output))
    # print(output.shape)

    # print("target:{}".format(target))
    # print(target.shape)
    
    output_ = output > 0.5
    target_ = target > 0.5 
    # print("output_:{}".format(output_))
    # print(output_.shape)

    # print("target_:{}".format(target_))
    # print(target_.shape)

    intersection = (output_ & target_).sum()  #交集
    union = (output_ | target_).sum()  #并集
    
    #TN=(~output_+2).sum()+intersection
    # TN=(~(output_ | target_)+2).sum()
    TN=((~output_) &(~ target_)).sum() 
    FP=((output_) &( ~target_)).sum() 
    TP=intersection
    FN=target_.sum()-intersection
    TNFP=(~target_).sum()
    TPFN=target_.sum()
    # print("accuracy_output:{}".format(output_))
    # print(output_.shape)
    # print("accuracy_target:{}".format(target_))
    # print(target_.shape)
    # print("Accuracy:tn:{};fp{};tp:{};fn:{}".format(TN,FP,TP,FN))
    # print('TNFP{};TPFN{}'.format(TNFP,TPFN))


    return (TN+TP + smooth) / \
        (TNFP+TP+FN+ smooth)







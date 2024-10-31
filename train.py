import argparse
import logging
import os
import sys
from datetime import datetime
import modules
import torch
from thop import profile

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd
from eval import eval_net
from eval_module import eval_module_net
import openpyxl
from modules import UNet
from modules import Att_UNet
from modules import R2_UNet
from modules import Att_R2_UNet
from modules import Logo_UNet

from modules.UNet_ASPP_model import *
from modules.UNet_ASPP_Att_model import *

from modules.UNet_CBAM_model import*
from modules.UNet_CBAM_ASPP_model import*
from modules.UNet_CBAM_ASPP2_model import*
from modules.UNet_CBAM_Res_ASPP_model import*

from modules.UNet_SE_model import *
from modules.UNet_Res_model import *
from modules.UNet_Res_SE_model import *
from modules.UNet_SE_ASPP_model import *
# from modules.axial_parts import*

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from utils.dataset import CarvanaDataset
from torch.utils.data import DataLoader, random_split

import eval_fun as fun
import result_show as show


dir_checkpoint = 'checkpoints/'
save_model='modules/save_modules/'

#设置随机种子，确保模型稳定可复现
seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)#为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed(seed)#为当前GPU设置随机种子；
torch.cuda.manual_seed_all(seed)#为多GPU设置随机种子；

# torch.backends.cudnn.benchmark=False
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.enabled=False

def train_net(net,
              net_name,
              datasets_name,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              test_percent=0.0,
              save_cp=False,
              mask_suffix='',
              img_scale=0.5,
              img_size=512):

    #只使用isic2018训练集读取数据，使用自己定义随机划分好的训练集train和验证集Validation，并加载数据
    transforme=transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
			])

    train = BasicDataset(dir_train, dir_train_mask, img_scale,'',img_size,transforme=None)
    val = BasicDataset(dir_val, dir_val_mask, img_scale,mask_suffix,img_size,transforme=None)
    n_train=int(len(train))
    n_val=int(len(val))
    n_test=0

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    #创建tensorboard，可视化。意为自动生成保存路径名称。注释（学习率，batchsize和图像尺寸）加在文件名后面。
    writer = SummaryWriter(comment=f'_{args.module}_{args.datasets}_SZ_{img_size}_LR_{lr}_BS_{batch_size}_WD{args.wd}_GN_{args.gn}_RT_{args.ratio}_Pool_{args.pool}')
    datatime=datetime.now().strftime("%Y%m%d_%H%M%S")
    savedir = save_model+"/{}_{}_{}_SZ_{}_LR_{}_BS_{}_WD{}_GN_{}_RT_{}_Pool_{}/".format(datatime,args.module,args.datasets,img_size,lr,batch_size,args.wd,args.gn,args.ratio,args.pool)
    
    global_step = 0 #全局步值
    

    logging.info(f'''Starting training:

        DateSets name: {args.datasets}
        Training size:   {n_train}#
        Validation size: {n_val}
        Test size: {n_test}  

        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Weighte Decay:   {args.wd}

        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:  {img_size}
        Images scaling:  {img_scale}
        Ratios:{args.ratio}
        GN:{args.gn}

    ''')

    # 计算量和模型参数量
    # input1 = torch.randn(4, 3, 256, 256) 
    # input1 = input1.to(device=device, dtype=torch.float32)
    # flops, params = profile(net, inputs=(input1, ))
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')

    #定义优化器和学习率调整
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)

    optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=args.wd)
    # scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.9)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-8)

    #定义损失函数
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()#多分类用的交叉熵损失函数，用这个 loss 前面不需要加 Softmax 层
    else:
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss() #二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数

    #训练集结果
    epoch_loss_train_list=[]
    #验证集结果
    epoch_loss_test_list=[]    
    dice_list,acc_list,jass_list,spe_list,pre_list=[],[],[],[],[]
    epoch_dice_list,epoch_acc_list,epoch_jaccard_list,epoch_spe_list,epoch_sen_list=[],[],[],[],[]

    best_step_dice=0

    #迭代训练
    best_dice=0
    for epoch in range(epochs):
        net.train()
        step_n=0 #记录每个epoch内进行的验证次数，最终验证次数为10，及step_n=[1,2,3,...10]       
        epoch_dice,epoch_jaccard,epoch_pre,epoch_acc,epoch_rec,epoch_sen,epoch_spe=0,0,0,0,0,0,0
        epoch_loss_train,epoch_loss_test= 0,0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                # print("img.shape:{}".format(imgs.shape))
                #如果通道数不匹配将输出语句提示
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                #print("true_mask.shape:{}".format(true_masks.shape))
                masks_pred = net(imgs)
                #print("mask_pred.shape:{}".format(masks_pred.shape))
                # true true_masks.squeeze(1)
                # print("mask_shape{}".format(masks_pred.shape))
                # print("target_shape{}".format(true_masks.shape))   
                loss = criterion(masks_pred, true_masks)
                epoch_loss_train += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                #通过set_description（左）和set_postfix（右）方法设置进度条左边和右边显示信息
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                #Clips gradient of an iterable of parameters at specified value.剪辑参数在指定值的可迭代对象的梯度。
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                
                #通过update方法可以控制每次进度条更新的进度
                pbar.update(imgs.shape[0])

                #信息记录
                global_step += 1
                #一个epoch内进行10次验证
                # if global_step % (n_train // (10 * batch_size)) == 0:
                if global_step % (n_train // (10 * batch_size)) == 0:
                    step_n+=1
                    #print('n_train:{}；batch_size:{};global:{};step_n:{}'.format(n_train,batch_size,global_step,step_n))
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)#报错！！！可能有未使用的定义层
                
                    #在验证集上评价指标结果的输出和存储：

                    #需修改eval_module_net文件，并对n_class分情况讨论，并添加logging.info和writer.add_scalar的语句进行显示和存储！！！
                    #val_score = eval_net(net, val_loader, device)
                    #scheduler.step(val_score)
                    # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    result = eval_module_net(net, val_loader, device,savedir,step_n,epoch,best_dice)###huan
                    # scheduler.step(result[0])#scheduler的更新官方说明应该放在epoch里，不是batch里
                    if result[0]>best_step_dice:
                        best_step_dice=result[0]
                        best_step_jass=result[2]
                        best_step_acc=result[10]
                        best_step_sen=result[6]
                        best_step_spe=result[7]
                        best_step=step_n
                        best_step_epoch=epoch+1

                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    if net.n_classes > 1:
                    # if classes > 1:
                        logging.info('Validation cross entropy: {}'.format(result[0]))#交叉熵
                        writer.add_scalar('Loss/test', result[0], global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(result[0]))#Dice
                        # writer.add_scalar('Dice/test', result[0], global_step)
                        # dice_coef_list.append(result[0])
                        # epoch_dice+=result[0]
                        # huan_define:
                        logging.info('Dice: {}'.format(result[1]))
                        writer.add_scalar('Dice/test', result[1], global_step)
                        epoch_dice+=result[1]

                        logging.info('Jass: {}'.format(result[2])) 
                        writer.add_scalar('Jaccard/test', result[2], global_step)
                        epoch_jaccard+=result[2]

                        # logging.info('VOE: {}'.format(result[3])) 
                        # writer.add_scalar('VOE/test', result[3], global_step)
                        # voe_list.append(result[3])

                        # # logging.info('RVD: {}'.format(result[4])) 
                        # writer.add_scalar('RVD/test', result[4], global_step)
                        # rvd_list.append(result[4])  

                        # # logging.info('IoU: {}'.format(result[5])) 
                        # # writer.add_scalar('IoU/test', result[5], global_step)

                        logging.info('Sensitivity: {}'.format(result[6])) 
                        writer.add_scalar('Sensitivity/test', result[6], global_step)
                        epoch_sen+=result[6]

                        logging.info('Specificity: {}'.format(result[7])) 
                        writer.add_scalar('Specificity/test', result[7], global_step)   
                        epoch_spe+=result[7]    

                        # logging.info('Pre: {}'.format(result[8])) 
                        # writer.add_scalar('Pre/test', result[8], global_step)
                        # epoch_pre+=result[8]  
                        
                        # logging.info('Recall: {}'.format(result[9])) 
                        # writer.add_scalar('Recall/test', result[9], global_step)
                        # epoch_rec+=result[9]      

                        logging.info('Accuracy: {}'.format(result[10])) 
                        writer.add_scalar('Accuracy/test', result[10], global_step) 
                        epoch_acc+=result[10]     

                        logging.info('val_loss: {}'.format(result[11])) 
                        writer.add_scalar('Loss/test', result[11], global_step) 
                        epoch_loss_test+=result[11]   
         
                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)  

                    logging.info('best_epoch{}_step{}:best_dice:{:.4f};best_jaccard:{:.4f};best_acc:{:.4f};best_sen:{:.4f};best_spe:{:.4f}'.format(best_step_epoch,best_step,best_step_dice,best_step_jass,best_step_acc,best_step_sen,best_step_spe))     
        
        #保存最佳模型（loss最小）
        if (epoch_dice/step_n)>best_dice:
            best_dice=epoch_dice/step_n
            best_epoch=epoch
            if not os.path.isdir(savedir):
                os.makedirs(savedir)
            torch.save(net.state_dict(),savedir + 'best_model.pth')
            logging.info(f'epoch{epoch + 1} best_model saved !')

        #记录每轮的评价指标结果
        logging.info('epoch{}_loss:{}'.format(epoch+1,epoch_loss_train/len(train_loader)))
        writer.add_scalar('Loss/train/epoch', epoch_loss_train/ len(train_loader),epoch)
        epoch_loss_train_list.append(epoch_loss_train/len(train_loader))

        writer.add_scalar('Loss/test/epoch', epoch_loss_test/ step_n,epoch+1)
        epoch_loss_test_list.append(epoch_loss_test/step_n)

        logging.info('epoch{}:'.format(epoch+1))

        writer.add_scalar('Dice/test/epoch', epoch_dice/step_n, epoch)
        epoch_dice_list.append(epoch_dice/step_n)
        logging.info('Dice:{}'.format(epoch_dice/step_n))

        writer.add_scalar('Jaccard/test/epoch', epoch_jaccard/ step_n,epoch)
        epoch_jaccard_list.append(epoch_jaccard/step_n)
        logging.info('Jaccard:{}'.format(epoch_jaccard/step_n))

        writer.add_scalar('Accuracy/test/epoch', epoch_acc/ step_n,epoch)
        epoch_acc_list.append(epoch_acc/step_n)
        logging.info('Accuracy:{}'.format(epoch_acc/step_n))

        writer.add_scalar('Sensitivity/test/epoch', epoch_sen/step_n,epoch)
        epoch_sen_list.append(epoch_sen/step_n)
        logging.info('Sensitivity:{}'.format(epoch_sen/step_n))

        writer.add_scalar('Specificity/test/epoch', epoch_spe/ step_n,epoch)
        epoch_spe_list.append(epoch_spe/step_n)
        logging.info('Specificity:{}'.format(epoch_spe/step_n))

        print('第{}个epoch的学习率为：{}'.format(epoch+1,optimizer.param_groups[0]['lr']))
        scheduler.step(epoch_dice/step_n)

        ############# Plot val curve
        # plt.figure()
        # plt.plot(epoch_acc_list, label='val acc', color='blue', linestyle='--')
        # plt.legend(loc='best')
        # plt.savefig(os.path.join(savedir, 'Validation_Accuracy.png'))

        # plt.figure()
        # plt.plot(epoch_dice_list, label='val dice', color='red', linestyle='--')
        # plt.legend(loc='best')
        # plt.savefig(os.path.join(savedir, 'Validation_Dice.png'))

        # plt.clf()
        # plt.close()
        # # plt.show()

        # plt.close('all')

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
        
        logging.info('best_epoch:{};best_dice:{:.4f};best_jaccard:{:.4f};best_acc:{:.4f};best_sen:{:.4f};best_spe:{:.4f}'.format(best_epoch,epoch_dice_list[best_epoch],epoch_jaccard_list[best_epoch],epoch_acc_list[best_epoch],epoch_sen_list[best_epoch],epoch_spe_list[best_epoch]))
        logging.info('best_epoch{}_step:{};best_dice:{:.4f};best_jaccard:{:.4f};best_acc:{:.4f};best_sen:{:.4f};best_spe:{:.4f}'.format(best_step_epoch,best_step,best_step_dice,best_step_jass,best_step_acc,best_step_sen,best_step_spe))

    logging.info(f'''End training:

    Network:{args.module}
    DateSets name: {args.datasets}
    Training size:   {n_train}#
    Validation size: {n_val}
    Test size: {n_test}  

    Epochs:          {epochs}
    Batch size:      {batch_size}
    Learning rate:   {lr}
    Weighte Decay:   {args.wd}

    Checkpoints:     {save_cp}
    Device:          {device.type}
    Images size:  {img_size}
    Images scaling:  {img_scale}
    Ratios:{args.ratio}
    GN:{args.gn}
''')

    #将数据写入excel表格
    output_excel = {'train_loss':[], 'test_loss':[], 'Dice':[], 'Jaccard':[], 'Accuracy':[], 'Sen':[], 'Spe':[]}
    output_excel['train_loss'] = epoch_loss_train_list
    output_excel['test_loss'] = epoch_loss_test_list
    output_excel['Dice'] = epoch_dice_list
    output_excel['Jaccard'] = epoch_jaccard_list
    output_excel['Accuracy'] = epoch_acc_list
    output_excel['Sen'] = epoch_sen_list
    output_excel['Spe'] = epoch_spe_list
    output = pd.DataFrame(output_excel)
    excel_dir=savedir+'epoch_data.xlsx'
    output.to_excel(excel_dir, index=False)
    #保存最后的模型
    torch.save(net.state_dict(), savedir+"final_model.pth")
    logging.info('final_model saved !')
    writer.close()


#参数设置
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--module', metavar='M', type=str,default='cbam_aspp',
                    help='model architecture: unet/att_unet/r2_unet/r2a_unet...')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-wd', '--weightdecay', metavar='WD', type=float, nargs='?', default=1e-8,
                        help='weight decay', dest='wd')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-p', '--pool', dest='pool', type=str, default=False,
                        help='if use AdaptiveAvgPool2d in aspp_block')
    parser.add_argument('-sp', '--save_cp', dest='save_cp', type=str, default=True,
                        help='if save checkpoint')
    parser.add_argument('-bi', '--bilinear', dest='bilinear', type=str, default=True,
                        help='if Upsample or ConvTranspose2d')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0,
                        help='Downscaling factor of the images')
    parser.add_argument('-sz', '--size', dest='size', type=int, default=256,
                        help='Size of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-t', '--test', dest='test', type=float, default=0.0,
                        help='Percent of the data that is used as test (0-100)')
    # parser.add_argument('-ch', '--channel', dest='channel', type=int, default=1,
    #                     help='if image is RGB,channels should be 3')             
    # parser.add_argument('-cl', '--class_type', dest='class_type', type=int, default=1,
    #                     help='The number of class types ') 
    parser.add_argument('-d', '--datasets', dest='datasets', type=str, default='isic2018',
                        help='DataSets:car/cell/lung/skin/ ... ')  
    parser.add_argument('-r', '--ratio', dest='ratio', nargs="+",type=int, default=[6,12,18],
                        help='radio of the images')  
    parser.add_argument('-re', '--reduce', dest='reduce',type=int, default=16,
                        help='reduce of the se block')  
    parser.add_argument('-gn', '--gn', dest='gn',type=int, default=64,
                        help='the group of GroupNormalization')  
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #`%(levelname)s`  表示日志级别名称   %(message)s`    表示日志内容
    #https://zhuanlan.zhihu.com/p/38781838
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}') #logging.info是print的功能，输出日志信息

   # Change here to adapt to your data
    # n_channels=3 for RGB images，n_channels=1 for graylevel images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N

    #根据数据集修改代码
    mask_suffix='' #默认后缀为空
    if args.datasets=='cell':
        datasets_name='cell'
        dir_img = 'data/cell/image/'
        dir_mask = 'data/cell/label/'
        channel=1
        class_type=1
    elif args.datasets=='car':
        datasets_name='car'
        dir_train = 'data/car/train/'
        dir_train_mask = 'data/car/train_masks/'
        mask_suffix='_mask'
        channel=1
        class_type=1

    elif args.datasets=='lung':#!!!数据集有问题，channel数有3有1，修改dataset文件，统一为1(L)通道或3(RGB)通道
        datasets_name='lung'
        dir_img = 'data/lung/images/'
        dir_mask = 'data/lung/masks/'
        # mask_suffix='_mask'
        channel=1
        class_type=1
        
    elif args.datasets=='lht':#数据集太大
        dir_img = 'data/lht/images/'
        dir_mask = 'data/lht/masks/'
        mask_suffix='_mask' ###改！！！！
        channel=1
        class_type=3

    elif args.datasets=='isic2017':#数据集太大，图片大小不一样，需统一尺寸（dataset.py）
        datasets_name='ISIC2017'
        dir_train = 'data/ISIC2017/Training_resize_seg/Images/'
        dir_train_mask = 'data/ISIC2017/Training_resize_seg/Annotation/'
        dir_val = 'data/ISIC2017/Validation_Data/Images/'
        dir_val_mask = 'data/ISIC2017/Validation_Data/Annotation/'
        mask_suffix='_segmentation' 
        channel=3
        class_type=1

    elif args.datasets=='isic2017_enhance':#数据集太大，图片大小不一样，需统一尺寸（dataset.py）
        datasets_name='ISIC2017'
        dir_train = 'data/ISIC2017/Training_enhance_seg/Images/'
        dir_train_mask = 'data/ISIC2017/Training_enhance_seg/Annotation/'
        dir_val = 'data/ISIC2017/Validation_Data/Images/'
        dir_val_mask = 'data/ISIC2017/Validation_Data/Annotation/'
        mask_suffix='_segmentation' 
        channel=3
        class_type=1

    elif args.datasets=='isic2017_crop':
        datasets_name='ISIC2017_crop'
        dir_train = 'data/ISIC2017/Training_crop_seg/Images/'
        dir_train_mask = 'data/ISIC2017/Training_crop_seg/Annotation/'
        dir_val = 'data/ISIC2017/Validation_crop_seg/Images/'
        dir_val_mask = 'data/ISIC2017/Validation_crop_seg/Annotation/'
        mask_suffix='' 
        channel=3
        class_type=1

    elif args.datasets=='isic2018':#数据集太大，图片大小不一样，需统一尺寸（dataset.py）
        datasets_name='ISIC2018'
        dir_train = 'data/ISIC2018/date_9_1_2/train/image/'
        dir_train_mask = 'data/ISIC2018/date_9_1_2/train/label/'
        dir_val = 'data/ISIC2018/date_9_1_2/val/image/'
        dir_val_mask = 'data/ISIC2018/date_9_1_2/val/label/'
        mask_suffix='_segmentation' 
        channel=3
        class_type=1    

    #根据输入的参数指定网络结构，默认UNet
    if args.module=='unet':
        net_name='unet'
        gn=args.gn
        bi=args.bilinear
        net = UNet(n_channels=channel, n_classes=class_type, bilinear=bi,gn=gn)
    elif args.module=='aspp_unet':
        net_name='assp_unet'
        ratio=args.ratio
        gn=args.gn
        bi=args.bilinear
        net = ASPP_UNet(n_channels=channel, n_classes=class_type, bilinear=bi,ratios=ratio,gn=gn,pool=args.pool)
    elif args.module=='att_aspp':
        net_name='aspp_att_unet'
        ratio=args.ratio
        gn=args.gn
        bi=args.bilinear
        net = ASPP_Att_UNet(n_channels=channel, n_classes=class_type, bilinear=bi,ratios=ratio,gn=gn,pool=args.pool)
    elif args.module=='se_unet':
        net_name='se_unet'
        re=args.reduce
        gn=args.gn
        bi=args.bilinear
        net = SE_UNet(n_channels=channel, n_classes=class_type, bilinear=bi,reduce=re,gn=gn)
    elif args.module=='res_unet':
        net_name='res_unet'
        gn=args.gn
        bi=args.bilinear
        net = ResUNet(n_channels=channel, n_classes=class_type, bilinear=bi,gn=gn)

    elif args.module=='res_se_unet':
        net_name='res_se_unet'
        re=args.reduce
        gn=args.gn
        bi=args.bilinear
        net = ResSeUNet(n_channels=channel, n_classes=class_type, bilinear=bi,reduce=re,gn=gn)

    elif args.module=='se_aspp':
        net_name='se_aspp_unet'
        ratio=args.ratio
        gn=args.gn
        bi=args.bilinear
        net = SE_ASPP_UNet(n_channels=channel, n_classes=class_type, bilinear=bi,ratios=ratio,gn=gn,pool=args.pool)
    
    elif args.module=='cbam':
        net_name='cbam_unet'
        ratio=args.ratio
        gn=args.gn
        bi=args.bilinear
        net = CBAM_UNet(n_channels=channel, n_classes=class_type, bilinear=bi,ratios=ratio,gn=gn,pool=args.pool)
    elif args.module=='cbam_aspp':
        net_name='cbam_aspp_unet'
        ratio=args.ratio
        gn=args.gn
        bi=args.bilinear
        net = CBAM_ASPP_UNet(n_channels=channel, n_classes=class_type, bilinear=bi,ratios=ratio,gn=gn,pool=args.pool)
    elif args.module=='cbam_aspp2':
        net_name='cbam_aspp_unet2'
        ratio=args.ratio
        gn=args.gn
        bi=args.bilinear
        net = CBAM_ASPP_UNet2(n_channels=channel, n_classes=class_type, bilinear=bi,ratios=ratio,gn=gn,pool=args.pool)
    elif args.module=='res_cbam_aspp':
        net_name='res_cbam_aspp'
        ratio=args.ratio
        gn=args.gn
        bi=args.bilinear
        net = ResCBAM_ASPP_UNet(n_channels=channel, n_classes=class_type, bilinear=bi,ratios=ratio,gn=gn,pool=args.pool)

    elif args.module=='att_unet':
        net_name='att_unet'
        net = Att_UNet(n_channels=channel, n_classes=class_type, bilinear=True)
    elif args.module=='r2_unet':
        net_name='r2_unet'
        net = R2_UNet(n_channels=channel, n_classes=class_type, bilinear=True)
    elif args.module=='r2a_unet':
        net_name='r2a_unet'
        net = Att_R2_UNet(n_channels=channel, n_classes=class_type, bilinear=True)
    elif args.module=='logo_unet':
        net_name='logo_unet'
        net = Logo_UNet(n_channels=channel, n_classes=class_type, bilinear=True)
    elif args.module=='axial_unet':
        net_name='axial_unet'
        img_size=args.size#256
        net = modules.Axial_model.ResAxialAttentionUNet(AxialBlock, [1, 2, 4, 1], s= 0.125, img_size=img_size,n_channels = channel,n_classes=class_type, bilinear=True)
        #注：此处的s与参数设置的scale含义不同。
    elif args.module=='gate_unet':
        net_name='gate_unet'
        img_size=args.size#256
        net = modules.Axial_Gated_model.ResAxialAttentionUNet(AxialBlock_dynamic, [1, 2, 4, 1], s= 0.125, img_size=img_size,n_channels = channel,n_classes=class_type, bilinear=True)
    elif args.module=='logo_unet':
        net_name='logo_unet'
        img_size=args.size#128
        net = modules.Axial_Logo_model.medt_net(AxialBlock,AxialBlock, [1, 2, 4, 1], s= 0.125, img_size=img_size,n_channels = channel,n_classes=class_type, bilinear=True)
    elif args.module=='medt_unet':
        net_name='medt_unet'
        img_size=args.size#128
        net = modules.Axial_Gated_Logo_Med_model .medt_net(AxialBlock_dynamic,AxialBlock_wopos, [1, 2, 4, 1], s= 0.125, img_size=img_size,n_channels = channel,n_classes=class_type, bilinear=True)        
    else:print('ERROR!! This module still not be define')

    # logging.info(f'Module_names:{args.module}\n'
    #             #  f'\tNetwork:\n'
    #              f'\t{net.n_channels} input channels\n'
    #              f'\t{net.n_classes} output channels (classes)\n'
    #              f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  net_name=net_name,
                  datasets_name=datasets_name,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  save_cp=args.save_cp,
                  mask_suffix=mask_suffix,
                  img_scale=args.scale,
                  img_size=args.size,
                  val_percent=args.val / 100,
                  test_percent=args.test/100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


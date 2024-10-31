import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from dice_loss import dice_coeff
from eval_fun import dice_coef
import eval_fun as fun
from torchvision import utils as vutils
import os
def eval_module_net2(net, loader, device,save_dir,epoch):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    #使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval
    # eval()时，框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值
    e=str(epoch+1)
    # imgdir=save_dir+"/epoch{}/test{}/".format(e,step_n)
    imgdir=save_dir+"/epoch{}/".format(e)
    n=0
    if not os.path.isdir(imgdir):
        os.makedirs(imgdir)
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    dice,jass,voe,rvd,asd,msd=0,0,0,0,0,0
    iou=0
    sen=0
    spe,pre,re,acc=0,0,0,0
    criterion = nn.BCEWithLogitsLoss()
    loss=0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            n+=1
            no=str(n)
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
            

            if net.n_classes > 1:
                pred=mask_pred
                tot += F.cross_entropy(pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                loss+=criterion(pred, true_masks).item()
                # print("val_loss(batch):{}".format(loss))
                pred = (pred > 0.5).float()
                true_masks = (true_masks > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()

                # print("output:{}".format(pred))
                # print("outputshape:{}".format(pred.shape))
                # print("target:{}".format(true_masks))
                # print("targetshape:{}".format(true_masks.shape))

                # #n_classes未做讨论
                # for i, p in enumerate(zip(pred, true_masks)):
                #     output=p[0]
                #     target=p[1]

                #     dice += fun.dice_coef(output, target)
                #     jass +=fun.jassard(output, target)
                #     voe +=fun.voe(output, target)
                #     rvd +=fun.rvd(output, target)

                #     iou=iou+fun.iou_score(output, target)

                #     sen=sen+fun.sensitivity(output, target)
                #     spe=spe+fun.specificity(output, target)
                #     pred=pre+fun.precision(output, target)
                #     re=re+fun.recall(output, target)
                #     acc=acc+fun.accuracy(output, target)
                # pbar.update()

###############################################################################
                #保存结果图片
                imgname=imgdir + 'batch'+no+'_img.png'
                maskname=imgdir + 'batch'+no+'_mask.png'
                predname=imgdir + 'batch'+no+'_pred.png'

                vutils.save_image(imgs, imgname)
                vutils.save_image(true_masks, maskname)
                vutils.save_image(pred, predname)
#############################################################################
                for i, p in enumerate(zip(pred, true_masks)):
                    pred=p[0]
                    true_masks=p[1]

                    dice += fun.dice_coef(pred, true_masks)
                    jass +=fun.jassard(pred, true_masks)
                    voe +=fun.voe(pred, true_masks)
                    rvd +=fun.rvd(pred, true_masks)
                    iou=iou+fun.iou_score(pred, true_masks)
                    sen=sen+fun.sensitivity(pred, true_masks)
                    spe=spe+fun.specificity(pred, true_masks)
                    pre=pre+fun.precision(pred, true_masks)
                    re=re+fun.recall(pred, true_masks)
                    acc=acc+fun.accuracy(pred, true_masks)

                
                # dice += fun.dice_coef(pred, true_masks)
                # jass +=fun.jassard(pred, true_masks)
                # voe +=fun.voe(pred, true_masks)
                # rvd +=fun.rvd(pred, true_masks)

                # iou=iou+fun.iou_score(pred, true_masks)

                # sen=sen+fun.sensitivity(pred, true_masks)
                # spe=spe+fun.specificity(pred, true_masks)
                # pre=pre+fun.precision(pred, true_masks)
                # re=re+fun.recall(pred, true_masks)
                # acc=acc+fun.accuracy(pred, true_masks)

                # loss=loss+ criterion(pred, true_masks).item()



    eval_result=[tot / n_val,dice/(n_val*4),jass/(n_val*4),voe/(n_val*4),rvd/(n_val*4),iou/(n_val*4),sen/(n_val*4),spe/(n_val*4),pre/(n_val*4),re/(n_val*4),acc/(n_val*4),loss/(n_val*4)]
    
    net.train()
    return eval_result
    #return tot / n_val,dice/n_val,iou/n_val


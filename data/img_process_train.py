import os
from PIL import Image
import numpy as np
from torchvision import transforms

#重置图片大小
def resize_bic(image, label):
    w, h = label.size
    x = 1600
    if (w > x) | (h > x):
        if w > h:
            y = int(h * x / w)
            out_im = image.resize((x, y), Image.BICUBIC)#双三次插值
            out_la = label.resize((x, y), Image.NEAREST)#最邻近插值
        else:
            y = int(w * x / h)
            out_im = image.resize((y, x), Image.BICUBIC)
            out_la = label.resize((y, x), Image.NEAREST)

    else:
        out_im = image
        out_la = label

    return out_im, out_la


#裁减（沿切点）
def resize_crop(image, label):
    #获取图像矩阵mask对应的矩形区域
    index = np.nonzero(label)
    img=np.array(image)
    lab=np.array(label)
    if len(index[0])!=0 and len(index[1])!=0:
        minx = np.min(index[1])
        maxx = np.max(index[1])
        miny = np.min(index[0])
        maxy = np.max(index[0])
        print(minx,maxx)
        out_im = img[miny:maxy, minx:maxx]
        out_la = lab[miny:maxy, minx:maxx]

        out_im = transforms.ToPILImage()(out_im)
        out_la = transforms.ToPILImage()(out_la)

    return out_im, out_la



#图片增强（水平、垂直、90/180/270）
def date_enhance(image, label,index):
    out_im0=image
    out_la0=label
    out_im0.save(path_enhance + index[:-4] + '.png')
    out_la0.save(path_gt_enhance + index[:-4] + '.png')
    out_im1 = image.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转
    out_la1 = label.transpose(Image.FLIP_LEFT_RIGHT)
    out_im1.save(path_enhance + index[:-4] + '_1.png')
    out_la1.save(path_gt_enhance + index[:-4] + '_1.png')

    out_im2 = image.transpose(Image.FLIP_TOP_BOTTOM)    #垂直翻转
    out_la2 = label.transpose(Image.FLIP_TOP_BOTTOM)
    out_im2.save(path_enhance + index[:-4] + '_2.png')
    out_la2.save(path_gt_enhance + index[:-4] + '_2.png')

    # out_im3 = image.rotate(90)       #90°顺时针翻转
    # out_la3 = label.rotate(90)
    # out_im3.save(path_enhance + index[:-4] + '_3.png')
    # out_la3.save(path_gt_enhance + index[:-4] + '_3.png')    

    out_im4 = image.rotate(180)       #180°顺时针翻转
    out_la4 = label.rotate(180)
    out_im4.save(path_enhance + index[:-4] + '_4.png')
    out_la4.save(path_gt_enhance + index[:-4] + '_4.png')
    
    # out_im5 = image.rotate(270)       #90°顺时针翻转
    # out_la5 = label.rotate(270)
    # out_im5.save(path_enhance + index[:-4] + '_5.png')
    # out_la5.save(path_gt_enhance + index[:-4] + '_5.png')    
    #out1.show()
    


##### Training
class_p = 'Training'

imagePathDir = os.listdir('/home/fzz/huaner/codes/MB-DCNN-master/dataset/data/ISIC-2017_'+class_p+'_Data/Images/')
imagePathDir.sort()
maskPathDir = os.listdir('/home/fzz/huaner/codes/MB-DCNN-master/dataset/data/ISIC-2017_'+class_p+'_Data/Annotation/')
maskPathDir.sort()

path_new = 'ISIC2017/'+class_p+'_resize_seg/Images/'
if not os.path.isdir(path_new):
    os.makedirs(path_new)

path_gt_new = 'ISIC2017/'+class_p+'_resize_seg/Annotation/'
if not os.path.isdir(path_gt_new):
    os.makedirs(path_gt_new)

path_crop = 'ISIC2017/'+class_p+'_crop_seg/Images/'
if not os.path.isdir(path_crop):
    os.makedirs(path_crop)

path_gt_crop = 'ISIC2017/'+class_p+'_crop_seg/Annotation/'
if not os.path.isdir(path_gt_crop):
    os.makedirs(path_gt_crop)

path_enhance = 'ISIC2017/Training_enhance_seg/Images/'
if not os.path.isdir(path_enhance):
    os.makedirs(path_enhance)    

path_gt_enhance = 'ISIC2017/Training_enhance_seg/Annotation/'
if not os.path.isdir(path_gt_enhance):
    os.makedirs(path_gt_enhance)

num = 0
label = []
train = []
for index in imagePathDir:
    print(num)
    # read img
    img = Image.open('/home/fzz/huaner/codes/MB-DCNN-master/dataset/data/ISIC-2017_'+class_p+'_Data/Images/'+index)
    mask = Image.open('/home/fzz/huaner/codes/MB-DCNN-master/dataset/data/ISIC-2017_'+class_p+'_Data/Annotation/'+index[:-4]+'_segmentation.png')
    
    # img_re, mask_re = resize_bic(img, mask)
    # img_re.save(path_new + index[:-4] + '.png')
    # mask_re.save(path_gt_new + index[:-4] + '.png')

    # img_cp, mask_cp = resize_crop(img,mask)
    # img_cp.save(path_crop + index[:-4] + '.png')
    # mask_cp.save(path_gt_crop + index[:-4] + '.png')

    date_enhance(img, mask,index)

    num = num + 1



from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance

class BasicDataset(Dataset):
    
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='_segmentation',img_size=256,transforme=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix ='_segmentation' 
        self.img_size=img_size
        self.transforme=transforme
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        #splitext(file)：分离文件名与拓展名，默认返回(fname,fextension)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)


    @classmethod
    def preprocess_img(cls, pil_img, scale):#对图片预处理
        w, h  = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        #pil_img = pil_img.convert('L') #统一图片为1通道
        #pil_img = pil_img.convert('RGB') #统一图片为3通道
        #print(pil_img.shape) #统一图片为1通道

        #滤波处理
        # pil_img=pil_img.filter(ImageFilter.DETAIL)#细节增强滤波
        # pil_img=pil_img.filter(ImageFilter.EDGE_ENHANCE)#边缘增强滤波
        # pil_img=pil_img.filter(ImageFilter.EDGE_ENHANCE_MORE)#深度边缘增强滤波
        # pil_img=pil_img.filter(ImageFilter.SHARPEN)#锐化滤波

        #增强处理（1为原图）
        # pil_img=ImageEnhance.Brightness(pil_img).enhance(1.2)#调整亮度
        # pil_img=ImageEnhance.Sharpness(pil_img).enhance(1.2)#锐度(2:完全锐化)
        # pil_img=ImageEnhance.Contrast(pil_img).enhance(1.5)#调整对比度

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    @classmethod
    def preprocess_mask(cls, pil_img, scale):#对图片掩膜预处理
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        #pil_img = pil_img.convert('RGB')
        #print(pil_img.shape)

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans
    
    def crop_img(self,img):
        #裁减图片，按长宽中较小的中心裁减，待确认是否正确
        # print('{}beforesize: {}'.format(idx,img.size))
        w,h=img.size
        if w<h:
            left=0
            upper=(h-w)/2
            right=w
            lower=w+(h-w)/2
        else:
            left=(w-h)/2
            upper=0
            right=h+(w-h)/2
            lower=h            
        img=img.crop((left, upper, right, lower))
        # mask=mask.crop((left, upper, right, lower))
        return img 

    def padding_img(self,img):
        w,h=img.size
        if w<h:
            left=(w-h)/2
            upper=0
            right=w+(h-w)/2
            lower=h            
        else:
            left=0
            upper=(h-w)/2
            right=w
            lower=h+(w-h)/2            
        img=img.crop((left, upper, right, lower))
        # mask=mask.crop((left, upper, right, lower))
        return img






    def __getitem__(self, i):
        idx = self.ids[i]
        # self.mask_suffix='_segmentation'
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        
        # print('idxmask_suffix{}'.format( self.mask_suffix ))
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        # print('{}beforesize: {}'.format(idx,img.size))

        # # 按短边裁减
        # img=self.crop_img(img)
        # mask=self.crop_img(mask)

        # # 按长边填充
        # img=self.padding_img(img)
        # mask=self.padding_img(mask)

        # print('{}cropsize:{}'.format(idx,img.size))

        #将图片大小统一（不统一的话，batch只能为1）
        mask = mask.resize((self.img_size, self.img_size))
        img = img.resize((self.img_size,self.img_size))
        # print('{}resize:{}'.format(idx,img.size))
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        if self.transforme is not None:
            img = self.transforme(img)
            mask=self.transforme(mask)
            
        img = self.preprocess_img(img, self.scale)
        # print("img.shape:{}".format(img.shape))
        mask = self.preprocess_mask(mask, self.scale)
        #print('mask.shape:{}'.format(mask.shape))
        # print('resize_end')

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')

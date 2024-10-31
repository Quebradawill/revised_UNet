import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
import eval_fun as fun
from modules import UNet
from modules import Att_UNet
from modules import R2_UNet
from modules import Att_R2_UNet
from modules import Logo_UNet
from modules.UNet_ASPP_model import *
from modules.UNet_CBAM_model import*
from modules.UNet_CBAM_ASPP_model import*
from modules.UNet_CBAM_ASPP2_model import*
from modules.UNet_CBAM_Res_ASPP_model import*
from modules.UNet_ASPP_Att_model import *
from modules.UNet_SE_model import *
from modules.UNet_ResSE_model import *
from modules.UNet_ResSE2_model import*
from modules.UNet_SE_ASPP_model import *

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess_img(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
    #                     help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()


def get_output_filenames(args,in_files):
    in_files = in_files
    out_files = []
    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            path_list=pathsplit[0].split('/')
            # print(path_list[-1])
            save_root_path='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/val/pre/'
            out_file=save_root_path+path_list[-1]+'_out'+pathsplit[1]
            out_files.append(out_file)
            # out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:

        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def get_input_file(root_path): 
    in_files=[]
    for dir_name in os.listdir(root_path):
        # exts = suffix.split(' ')
        #获取目录或文件的路径
        file_path = os.path.join(root_path,dir_name)
        in_files.append(file_path)
        #判断路径为文件还是路径
    return in_files


def get_file_num(root_path): 
    files_num=[]
    for dir_name in os.listdir(root_path):
        #获取目录或文件的路径
        file_path = os.path.join(root_path,dir_name)       
        pathsplit = os.path.splitext(file_path)
        path_list=pathsplit[0].split('/')
        files_num.append(path_list[-1])

    return files_num

def get_results():
    data_path='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/val'
    image='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/val/image'
    file_names=get_file_num(image)#获取验证集中图片名字
# print(file_names)
    dice=0
    acc=0
    for i in range (len(file_names)):
        image_path=data_path+'/image/'+file_names[i]+'.jpg'
        pre_path=data_path+'/pre/'+file_names[i]+'_out.jpg'
        label_path=data_path+'/label/'+file_names[i]+'_segmentation.png'

        scale_factor=1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image = Image.open(image_path)
        image = image.resize((256, 256))
        label = Image.open(label_path)
        label = label.resize((256, 256))
        pre=Image.open(pre_path)
        pre = pre.resize((256, 256))

        img = torch.from_numpy(BasicDataset.preprocess_img(image, scale_factor))
        #print("shape_before:{}".format(img.shape))
        img = img.unsqueeze(0)
        #print("shape_after:{}".format(img.shape))
        img = img.to(device=device, dtype=torch.float32)

        label = torch.from_numpy(BasicDataset.preprocess_mask(label, scale_factor))
        label = label.unsqueeze(0)
        label = label.to(device=device, dtype=torch.float32)

        pre = torch.from_numpy(BasicDataset.preprocess_mask(pre, scale_factor))
        pre = pre.unsqueeze(0)
        pre = pre.to(device=device, dtype=torch.float32)

        print('{}:Dice:{} ACC:{}'.format(file_names[i],fun.dice_coef(pre, label),fun.accuracy(pre, label)))

        dice += fun.dice_coef(pre, label)
        acc=acc+fun.accuracy(pre, label)
    print('Dice:{} ACC:{}'.format(dice/len(file_names),acc/len(file_names) ))
    return dice/len(file_names),acc/len(file_names)        

if __name__ == "__main__":
    args = get_args()
    # in_files = args.input
    image_path='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/val/image'
    in_files=get_input_file(image_path)

    # print(in_files)
    out_files = get_output_filenames(args,in_files)
    # print(out_files)
    net = CBAM_ASPP_UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        img = img.resize((256, 256))

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
        
    get_results()
        
        

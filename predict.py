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


# def predict_img(net,
#                 full_img,
#                 device,
#                 scale_factor=1,
#                 out_threshold=0.5):
#     net.eval()

#     img = torch.from_numpy(BasicDataset.preprocess_img(full_img, scale_factor))

#     img = img.unsqueeze(0)
#     img = img.to(device=device, dtype=torch.float32)

#     with torch.no_grad():
#         output = net(img)

#         if net.n_classes > 1:
#             probs = F.softmax(output, dim=1)
#         else:
#             probs = torch.sigmoid(output)

#         probs = probs.squeeze(0)

#         tf = transforms.Compose(
#             [
#                 transforms.ToPILImage(),
#                 transforms.Resize(full_img.size[1]),
#                 transforms.ToTensor()
#             ]
#         )

#         probs = tf(probs.cpu())
#         full_mask = probs.squeeze().cpu().numpy()

#     return full_mask > out_threshold


# def get_args():
#     parser = argparse.ArgumentParser(description='Predict masks from input images',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--model', '-m', default='MODEL.pth',
#                         metavar='FILE',
#                         help="Specify the file in which the model is stored")
#     parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
#                         help='filenames of input images', required=True)

#     parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
#                         help='Filenames of ouput images')
#     parser.add_argument('--viz', '-v', action='store_true',
#                         help="Visualize the images as they are processed",
#                         default=False)
#     parser.add_argument('--no-save', '-n', action='store_true',
#                         help="Do not save the output masks",
#                         default=False)
#     parser.add_argument('--mask-threshold', '-t', type=float,
#                         help="Minimum probability value to consider a mask pixel white",
#                         default=0.5)
#     parser.add_argument('--scale', '-s', type=float,
#                         help="Scale factor for the input images",
#                         default=1)

#     return parser.parse_args()


# def get_output_filenames(args):
#     in_files = args.input
#     out_files = []

#     if not args.output:
#         for f in in_files:
#             pathsplit = os.path.splitext(f)
#             out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
#     elif len(in_files) != len(args.output):
#         logging.error("Input files and output files are not of the same length")
#         raise SystemExit()
#     else:
#         out_files = args.output

#     return out_files


# def mask_to_image(mask):
#     return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    # args = get_args()
    # in_files = args.input
    # out_files = get_output_filenames(args)

    net = CBAM_ASPP_UNet(n_channels=3, n_classes=1,bilinear=True,ratios=[6,12,18],gn=64,pool=False)
    model='/home/fzz/huaner/codes/Pytorch-UNet-master/modules/save_modules/20221202_142131_cbam_aspp_isic2017_enhance_SZ_256_LR_0.0001_BS_4_WD1e-08_GN_64_RT_[6, 12, 18]_Pool_False/best_model.pth'

    logging.info("Loading model {}".format(model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))

    print("Model loaded")

    logging.info("Model loaded !")

#     for i, fn in enumerate(in_files):
#         logging.info("\nPredicting image {} ...".format(fn))

#         img = Image.open(fn)
#         img = img.resize((256, 256))

#         mask = predict_img(net=net,
#                            full_img=img,
#                            scale_factor=args.scale,
#                            out_threshold=args.mask_threshold,
#                            device=device)

#         if not args.no_save:
#             out_fn = out_files[i]
#             result = mask_to_image(mask)
#             result.save(out_files[i])

#             # logging.info("Mask saved to {}".format(o
#             # data_path='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/val'
#     image='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/val/image'
#     file_names=get_file_num(image)#获取验证集中图片名字
# # print(file_names)
#     dice=0
#     acc=0
#     for i in range (len(file_names)):
#         image_path=data_path+'/image/'+file_names[i]+'.jpg'
#         pre_path=data_path+'/pre/'+file_names[i]+'_out.jpg'
#         label_path=data_path+'/label/'+file_names[i]+'_segmentation.png'    
#         data_path='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/val'
#     image='/home/fzz/huaner/codes/Pytorch-UNet-master/data/ISIC2018/val/image'
#     file_names=get_file_num(image)#获取验证集中图片名字
# # print(file_names)
#     dice=0
#     acc=0
#     for i in range (len(file_names)):
#         image_path=data_path+'/image/'+file_names[i]+'.jpg'
#         pre_path=data_path+'/pre/'+file_names[i]+'_out.jpg'
#         label_path=data_path+'/label/'+file_names[i]+'_segmentation.png't_files[i]))

#         if args.viz:
#             logging.info("Visualizing results for image {}, close to continue ...".format(fn))
#             plot_img_and_mask(img, mask)

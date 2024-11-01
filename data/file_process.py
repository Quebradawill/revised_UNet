
# #删除掉image中有，mask中没有的数据
# import os
# import shutil

# images_list = []
# masks_list=[]
# for iroot, idirs, ifiles in os.walk('/home/fzz/huaner/codes/Pytorch-UNet-master/data/lung/images'):
#     images_list.append(ifiles)
#     # print(images_list)
# for mroot, mdirs, mfiles in os.walk('/home/fzz/huaner/codes/Pytorch-UNet-master/data/lung/masks'):
#     masks_list.append(mfiles)
# for iname in ifiles:
#     path=os.path.join(mroot,iname)
#     b=os.path.exists(path)
#     if b==0:
#         os.remove(os.path.join(iroot,iname))



#     #print(root)
#     #print(root) #当前目录路径
#         # print(dirs) #当前路径下所有子目录
#         # print(files) #当前路径下所有非目录子文件
# # print(files)

# # for name in files:
# #     if name == audio_name:
# #         os.remove(os.path.join(root, name))
# #         break
# b = os.path.exists #判断是否存在


# #删除指定文件后缀名的文件
# import os
# import glob

# path ='/home/fzz/huaner/codes/Pytorch-UNet-master/data/skin/image'
# for infile in glob.glob(os.path.join(path, '*.png')):
#      os.remove(infile)



#批量将bmp文件转换为PNG格式
import os
from PIL import Image

# bmp 转换为jpg
def bmpToJpg(file_path):
    for fileName in os.listdir(file_path):
        # print(fileName)
        newFileName = fileName[0:fileName.find("_")]+".jpg"
        print(newFileName)
        im = Image.open(file_path+"\\"+fileName)
        im.save(file_path+"\\"+newFileName)


# 删除原来的位图
def deleteImages(file_path, imageFormat):
    command = "del "+file_path+"\\*."+imageFormat
    os.system(command)


def main():
    file_path = "D:\\VideoPhotos"
    bmpToJpg(file_path)
    deleteImages(file_path, "bmp")


if __name__ == '__main__':
    main()

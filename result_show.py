from datetime import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# plot_reluts1(dice_list,voe_list,rvd_list,jass_list)
# plot_reluts2(spe_list,pre_list,acc_list,re_list)
dir_results='results/'
# datasets=
# module=
def plot_reluts1(list1,list2,list3,list4,step_n,lr,epoch,scale,batch_size,module,datasets):
    x1=range(0, step_n)
    y1=list1
    plt.subplot(221)
    plt.plot(x1,y1,'g-',label='Dice')
    # plt.ylabel('dice')
    # plt.xlabel('global_step')
    #plt.title(args.arch)
    plt.legend(loc='lower right')

    x2=range(0, step_n)
    y2=list2
    plt.subplot(222)
    plt.plot(x2,y2,'r-',label='VOE')
    # plt.ylabel('accuracy')
    # plt.xlabel('global_step')
    # plt.title(args.arch)
    plt.legend(loc='upper right')

    x3=range(0, step_n)
    y3=list3
    plt.subplot(223)
    plt.plot(x3,y3,'g-',label='RVD')
    # plt.ylabel('loss')
    # plt.xlabel('global_step')
    # plt.title('loss vs. epoches')
    plt.legend(loc='lower right')

    x4=range(0, step_n)
    y4=list4  
    plt.subplot(224)
    plt.plot(x4,y4,'r-',label='Jassward Index')
    # plt.ylabel('loss')
    # plt.xlabel('global_step')
    # plt.title('loss vs. epoches')
    plt.legend(loc='lower right')

    datatime=datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath=os.path.join(dir_results,datasets,module,
    r'result1_lr{}_e{}_s{}_b_{}_{}.jpg').format(lr,epoch,scale,batch_size,datatime)

    # if not os.path.exists(filepath):
    #     os.makedirs(filepath)

    plt.savefig(filepath)
    
    plt.show()

def plot_reluts2(list1,list2,list3,list4,step_n,lr,epoch,scale,batch_size,module,datasets):
    x1=range(0, step_n)
    y1=list1
    plt.subplot(221)
    plt.plot(x1,y1,'g-',label='Speficitity')
    plt.legend(loc='lower right')

    x2=range(0, step_n)
    y2=list2
    plt.subplot(222)
    plt.plot(x2,y2,'r-',label='Precision')
    plt.legend(loc='lower right')

    x3=range(0, step_n)
    y3=list3
    plt.subplot(223)
    plt.plot(x3,y3,'g-',label='Accuracy')
    plt.legend(loc='lower right')

    x4=range(0, step_n)
    y4=list4  
    plt.subplot(224)
    plt.plot(x4,y4,'r-',label='Recall')
    plt.legend(loc='lower right')

    datatime=datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath=os.path.join(dir_results,datasets,module,
    r'result2_lr{}_e{}_s{}_b_{}_{}.jpg').format(lr,epoch,scale,batch_size,datatime)
    # if not os.path.exists(filepath):
    #     os.makedirs(filepath)
    plt.savefig(filepath)
    plt.show()
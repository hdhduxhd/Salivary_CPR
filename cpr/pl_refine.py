import torch
import torchvision


import argparse
import importlib
import numpy as np
import random

from torch.utils.data import DataLoader
import scipy.misc
import torch.nn.functional as F
import os.path
import imageio
import sys
sys.path[0]='/kaggle/working/CPR/cpr'
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from torchvision import transforms
import matplotlib.pyplot as plt
from utils.metrics import dice_coefficient_numpy
import math
from PIL import Image

import networks.deeplabv3 as netd

import wandb



os.environ['CUDA_VISIBLE_DEVICES'] = '2'
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default='/kaggle/input/checkpoint-best/sim_learn_D2.pth.tar')
    parser.add_argument("--num_workers", default=0, type=int)#8
    #parser.add_argument("--alpha", default=16, type=int)
    parser.add_argument("--out_rw", type=str, default='./temp')
    parser.add_argument("--beta", default=2, type=int)#8
    parser.add_argument("--logt", default=2, type=int)#8
    parser.add_argument('--dataset', type=str, default='south')
    parser.add_argument('--data-dir', default='/kaggle/input/dataset')
    parser.add_argument('--out-stride',type=int,default=16)
    parser.add_argument('--sync-bn',type=bool,default=True)
    parser.add_argument('--freeze-bn',type=bool,default=False)
    parser.add_argument('--pseudo', type=str, default='/kaggle/input/checkpoint-best/pseudolabel_south.npz')
    parser.add_argument('--radius',type=int,default=4)
    parser.add_argument('-g', '--gpu', type=int, default=0)

    args = parser.parse_args()
    radius = args.radius
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    model = netd.DeepLab(num_classes=1, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn, image_res=512, radius=radius)
    
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, args.weights))
    checkpoint = torch.load(args.weights)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(model.predefined_featuresize)
    model.eval()

    composed_transforms_train = transforms.Compose([
        
        tr.Resize2(512,None,32,512),#512,None,32,512 #None,None,50,None      
        tr.Normalize_tf2(),
        tr.ToTensor2()
    ])

    # 设置随机种子以确保结果可重复
    random.seed(42)
    db = DL.FundusSegmentation_wprob(base_dir=args.data_dir, dataset=args.dataset, transform=composed_transforms_train, pseudo=args.pseudo)
    # 计算训练集和测试集的大小
    train_ratio = 0.7
    train_size = int(train_ratio * len(db))
    test_size = len(db) - train_size
    # 创建索引列表
    indices = list(range(len(db)))
    # 随机打乱索引
    random.shuffle(indices)
    # 根据打乱后的索引进行固定划分
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    # 创建训练集和测试集
    db_train = torch.utils.data.dataset.Subset(db, train_indices)
    db_test = torch.utils.data.dataset.Subset(db, test_indices)

    infer_data_loader = DataLoader(db_train, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    dice_before_cup = 0
    dice_after_cup = 0
    pseudo_label_dic = {}
    prob_dic = {}
    for iter, (img, _, name, prob, gt) in enumerate(infer_data_loader):

        name = name[0]
        #print(name)

        orig_shape = img.shape

        prob = prob.unsqueeze(1)
        prob_upsample = F.interpolate(prob, size=(img.shape[2], img.shape[3]), mode='bilinear')
        #prob_upsample = prob_upsample.squeeze(0)
        prob_upsample = (prob_upsample>0.75).float()
        # print("prob_upsample[:,0].shape:",prob_upsample[:,0].shape)
        # print("gt[:,0].shape:",gt[:,0].shape)
        
        dice_prob_cup = dice_coefficient_numpy(prob_upsample[:,0], gt[:,0])
        
        dice_before_cup += dice_prob_cup
        
        dheight = int(np.ceil(img.shape[2]/16))
        dwidth = int(np.ceil(img.shape[3]/16))

        cam = prob

        with torch.no_grad():
            _, _, _, aff_cup = model.forward(img.cuda(), True)
            aff_mat_cup = torch.pow(aff_cup, args.beta)

            trans_mat_cup = aff_mat_cup / torch.sum(aff_mat_cup, dim=0, keepdim=True)

            for _ in range(args.logt):
                trans_mat_cup = torch.matmul(trans_mat_cup, trans_mat_cup)

            cam_vec_cup = cam[:,0].view(1,-1)

            # print("cam_vec_cup.shape:",cam_vec_cup.shape)
            # print("trans_mat_cup.shape:",trans_mat_cup.shape)
            cam_rw_cup = torch.matmul(cam_vec_cup.cuda(), trans_mat_cup)

            cam_rw_cup = cam_rw_cup.view(1, 1, dheight, dwidth)

            cam_rw_save_cup = torch.nn.Upsample((512, 512), mode='bilinear')(cam_rw_cup)
            cam_rw = (cam_rw_save_cup[0,0]/torch.max(cam_rw_save_cup[0,0])).unsqueeze(0)
            # cam_rw = torch.stack((cam_rw_save_cup[0,0]/torch.max(cam_rw_save_cup[0,0]), cam_rw_save_disc[0,0]/torch.max(cam_rw_save_disc[0,0])))#/torch.max(cam_rw_save_cup[0,0])#/torch.max(cam_rw_save_disc[0,0])

            prob_dic[name] = cam_rw.detach().cpu().numpy()
            pseudo_label_dic[name] = (cam_rw>0.75).long().detach().cpu().numpy()
            

            cam_rw_save_cup = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw_cup)
            cam_rw = (cam_rw_save_cup[0,0]/torch.max(cam_rw_save_cup[0,0])).unsqueeze(0)
            # cam_rw = torch.stack((cam_rw_save_cup[0,0]/torch.max(cam_rw_save_cup[0,0]), cam_rw_save_disc[0,0]/torch.max(cam_rw_save_disc[0,0])))#/torch.max(cam_rw_save_cup[0,0])#/torch.max(cam_rw_save_disc[0,0])
            

            pseudo_label_rw = (cam_rw>0.75).long().detach().cpu().numpy()###(0.75*torch.max(cam_rw_save))

            plt.subplot(1, 4, 4,title='after')
            plt.imshow(pseudo_label_rw[0])


            dice_cam_rw_cup = dice_coefficient_numpy(np.expand_dims(pseudo_label_rw[0],0), gt[:,0])
            dice_after_cup += dice_cam_rw_cup
            
    dice_before_cup /= len(infer_data_loader)
    dice_after_cup /= len(infer_data_loader)    

    print('%.4f,%.4f'%(dice_before_cup,dice_after_cup))

    if not os.path.exists('./log'):
        os.mkdir('./log')

    if args.dataset=="south":
        np.savez('./log/pseudolabel_south_new', pseudo_label_dic, prob_dic)

    if args.dataset=="north":
        np.savez('./log/pseudolabel_north_new', pseudo_label_dic, prob_dic)

    if args.dataset=="west":
        np.savez('./log/pseudolabel_west_new', pseudo_label_dic, prob_dic)        

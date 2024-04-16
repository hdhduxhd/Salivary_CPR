import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

import os
import os.path as osp

import sys
sys.path[0]='/kaggle/working/Salivary_CPR'
import argparse
from torchvision import transforms
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr

from MFSAN.mfsan import MFSAN
from utils.metrics import *
import wandb

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
parser.add_argument('--datasetS1', type=str, default='west', help='/kaggle/input/dataset/west')
parser.add_argument('--datasetS2', type=str, default='south', help='/kaggle/input/dataset/south')
parser.add_argument('--datasetT', type=str, default='north', help='/kaggle/input/dataset/north')
parser.add_argument('--batch-size', type=int, default=8, help='batch size for training the model')
parser.add_argument('--data-dir', default='/kaggle/input/dataset', help='data root path')
parser.add_argument('--num_iters', type=int, default=1000, help='number of epoch')
parser.add_argument('--lr',  type=float, default=0.001, help='learning rate of model')
parser.add_argument('--log_interval',  type=int, default=10, help='the interval of log train loss')
parser.add_argument('--test_interval',  type=int, default=10, help='the interval of test to get the value of dice')
parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
parser.add_argument('--wandb_project_name', type=str, default='Salivary_CPR_MFSAN', help='specify wandb project name')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. dataset
composed_transforms_tr = transforms.Compose([
    #tr.Resize(512),###
    tr.RandomScaleCrop(512),
    tr.RandomRotate(),
    tr.RandomFlip(),
    tr.elastic_transform(),
    tr.add_salt_pepper_noise(),
    tr.adjust_light(),
    tr.eraser(),
    tr.Normalize_tf(),
    tr.ToTensor()
])

composed_transforms_ts = transforms.Compose([
    # tr.RandomCrop(512),
    tr.Resize(512),
    tr.Normalize_tf(),
    tr.ToTensor()
])

train_ratio = 0.8

domain_S1 = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetS1, transform=composed_transforms_tr)
train_size = int(train_ratio * len(domain_S1))
test_size = len(domain_S1) - train_size
domain_S1, domain_val1 = torch.utils.data.random_split(domain_S1, [train_size, test_size])
domain_loaderS1 = DataLoader(domain_S1, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
domain_iterS1 = iter(domain_loaderS1)
domain_val1.dataset.transform = composed_transforms_ts
domain_loader_val1 = DataLoader(domain_val1, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

domain_S2 = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetS2, transform=composed_transforms_tr)
train_size = int(train_ratio * len(domain_S2))
test_size = len(domain_S2) - train_size
domain_S2, domain_val2 = torch.utils.data.random_split(domain_S2, [train_size, test_size])
domain_loaderS2 = DataLoader(domain_S2, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
domain_iterS2 = iter(domain_loaderS2)
domain_val2.dataset.transform = composed_transforms_ts
domain_loader_val2 = DataLoader(domain_val2, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

domain_T = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetT, transform=composed_transforms_tr)
train_size = int(train_ratio * len(domain_T))
test_size = len(domain_T) - train_size
domain_T, domain_valT = torch.utils.data.random_split(domain_T, [train_size, test_size])
domain_loaderT = DataLoader(domain_T, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
domain_iterT = iter(domain_loaderT)
domain_valT.dataset.transform = composed_transforms_ts
domain_loader_valT = DataLoader(domain_valT, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

if args.use_wandb:
    wandb_run = wandb.init(project=args.wandb_project_name, name="test") if not wandb.run else wandb.run

model = MFSAN(img_size=512).to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# 训练ResU-net模型
max_source_dice = 0
max_target_dice = 0
for i in range(1, args.num_iters+1):
    model.train()  # 设置为训练模式

    #S1 & T
    try:
        sampleS = next(domain_iterS1)
    except Exception as err:
        domain_iterS1 = iter(domain_loaderS1)
        sampleS = next(domain_iterS1)
    try:
        sampleT = next(domain_iterT)
    except Exception as err:
        domain_iterT = iter(domain_loaderT)
        sampleT = next(domain_iterT)
    
    source_data, source_label = sampleS['image'].to(device), sampleS['map'].to(device)
    target_data = sampleT['image'].to(device)

    optimizer.zero_grad()
    mse_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, mark=1)
    gamma = 2 / (1 + math.exp(-10 * (i) / (args.num_iters))) - 1
    loss = mse_loss + gamma * (mmd_loss + l1_loss)
    loss.backward()
    optimizer.step()

    if i==1 or i % args.log_interval == 0:
        print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(i, 100. * i / args.num_iters, loss.item(), mse_loss.item(), mmd_loss.item(), l1_loss.item()))
        if args.use_wandb:
            wandb_run.log({"source1 loss":loss.item()})

    #S2 & T
    try:
        sampleS = next(domain_iterS2)
    except Exception as err:
        domain_iterS2 = iter(domain_loaderS2)
        sampleS = next(domain_iterS2)
    try:
        sampleT = next(domain_iterT)
    except Exception as err:
        domain_iterT = iter(domain_loaderT)
        sampleT = next(domain_iterT)
    
    source_data, source_label = sampleS['image'].to(device), sampleS['map'].to(device)
    target_data = sampleT['image'].to(device)

    optimizer.zero_grad()
    mse_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, mark=2)
    gamma = 2 / (1 + math.exp(-10 * (i) / (args.num_iters))) - 1
    loss = mse_loss + gamma * (mmd_loss + l1_loss)
    loss.backward()
    optimizer.step()

    if i==1 or i % args.log_interval == 0:
        print('Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(i, 100. * i / args.num_iters, loss.item(), mse_loss.item(), mmd_loss.item(), l1_loss.item()))
        if args.use_wandb:
            wandb_run.log({"source2 loss":loss.item()})
    
    #test
    if i==1 or i % args.test_interval == 0:
        mean_dice = 0
        model.eval()  # 设置为评估模式
        #S1
        total_dice = 0
        with torch.no_grad():
            for sample in domain_loader_val1:
                inputs = sample['image'].to(device)
                targets = sample['map'].to(device)
                
                outputs, _ = model(inputs, mark=3)
                dice = dice_coeff(outputs, targets)
                total_dice += dice.item()
        print('Iter [{}/{}], Source1 Test Dice: {:.4f}'.format(i, args.num_iters, total_dice/len(domain_loader_val1.dataset)))
        if args.use_wandb:
            wandb_run.log({"source1 dice":total_dice/len(domain_loader_val1.dataset)})
        mean_dice += total_dice/len(domain_loader_val1.dataset)
        #S2
        total_dice = 0
        with torch.no_grad():
            for sample in domain_loader_val2:
                inputs = sample['image'].to(device)
                targets = sample['map'].to(device)
                
                _, outputs = model(inputs, mark=3)
                dice = dice_coeff(outputs, targets)
                total_dice += dice.item()
        print('Iter [{}/{}], Source2 Test Dice: {:.4f}'.format(i, args.num_iters, total_dice/len(domain_loader_val2.dataset)))
        if args.use_wandb:
            wandb_run.log({"source2 dice":total_dice/len(domain_loader_val2.dataset)})
        mean_dice += total_dice/len(domain_loader_val2.dataset)
        mean_dice /= 2
        if mean_dice > max_source_dice:
            max_source_dice = mean_dice
            torch.save(model.state_dict(), "./best_model_source_iter{}.pth".format(i))
        #T
        total_dice = 0
        with torch.no_grad():
            for sample in domain_loader_valT:
                inputs = sample['image'].to(device)
                targets = sample['map'].to(device)
                
                pred1, pred2 = model(inputs, mark=3)
                outputs = (pred1 + pred2) / 2
                dice = dice_coeff(outputs, targets)
                total_dice += dice.item()
        print('Iter [{}/{}], Target Test Dice: {:.4f}'.format(i, args.num_iters, total_dice/len(domain_loader_valT.dataset)))
        if args.use_wandb:
            wandb_run.log({"target dice":total_dice/len(domain_loader_valT.dataset)})
        if total_dice/len(domain_loader_valT.dataset) > max_target_dice:
            max_target_dice = total_dice/len(domain_loader_valT.dataset)
            torch.save(model.state_dict(), "./best_model_target_iter{}.pth".format(i))

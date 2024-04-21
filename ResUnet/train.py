import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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

from ResUnet.res_unet import resnet50
from utils.metrics import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
parser.add_argument('--datasetS', type=str, default='west', help='test folder id contain images to test')#default='Domain4'
parser.add_argument('--datasetT', type=str, default='north', help='/kaggle/input/dataset/north')
parser.add_argument('--batch-size', type=int, default=8, help='batch size for training the model')
parser.add_argument('--data-dir', default='/kaggle/input/dataset', help='data root path')
parser.add_argument('--num_epochs', type=int, default=20, help='number of epoch')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. dataset
composed_transforms_tr = transforms.Compose([
    #tr.Resize(512),###
    tr.RandomScaleCrop(128),
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
    tr.Resize(128),
    tr.Normalize_tf(),
    tr.ToTensor()
])

domain = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetS, transform=composed_transforms_tr)
train_ratio = 0.7
train_size = int(train_ratio * len(domain))
test_size = len(domain) - train_size
domain_S, domain_val = torch.utils.data.random_split(domain, [train_size, test_size])
    
domain_loaderS = DataLoader(domain_S, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
domain_T = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetT, transform=composed_transforms_ts)
domain_loaderT = DataLoader(domain_T, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

domain_val.dataset.transform = composed_transforms_ts
# domain_val = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetS, split='test/ROIs', transform=composed_transforms_ts)
domain_loader_val = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

model = resnet50(3, 1, True).to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 训练ResU-net模型

max_dice = 0
for epoch in range(args.num_epochs):
    total_loss = 0

    model.train()  # 设置为训练模式
    
    for sample in domain_loaderS:
        inputs = sample['image'].to(device)
        targets = sample['map'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch+1, args.num_epochs, total_loss/len(domain_loaderS)))

    model.eval()  # 设置为评估模式
    total_dice = 0
    with torch.no_grad():
        for sample in domain_loader_val:
            inputs = sample['image'].to(device)
            targets = sample['map'].to(device)
            
            outputs = model(inputs)
            dice = dice_coeff(outputs, targets)
            total_dice += dice.item()
    
    print('Epoch [{}/{}], Source Test Dice: {:.4f}'.format(epoch+1, args.num_epochs, total_dice/len(domain_loader_val.dataset)))
    # if total_dice > max_dice:
    #     max_dice = total_dice
    #     torch.save(model.state_dict(), "./best_model_source.pth")

    total_dice = 0
    jaccard, accuracy, sensitivity, specificity = 0, 0, 0, 0
    dice_binary = 0
    jaccard_binary = 0
    with torch.no_grad():
        for sample in domain_loaderT:
            inputs = sample['image'].to(device)
            targets = sample['map'].to(device)
            
            outputs = model(inputs)
            dice = dice_coeff(outputs, targets)
            total_dice += dice.item()
            jaccard += jaccard_coeff(outputs, targets).item()
            accuracy += pixel_accuracy(outputs, targets)
            sensitivity += pixel_sensitivity(outputs, targets)
            specificity += pixel_specificity(outputs, targets)

            outputs = (outputs > 0.5).float()
            dice_binary += dice_coeff(outputs, targets).item()
            jaccard_binary += jaccard_coeff(outputs, targets).item()


    if total_dice > max_dice:
        max_dice = total_dice
        torch.save(model.state_dict(), "./best_model_target.pth")
        print('Epoch [{}/{}], Target Test Dice: {:.4f}, Jaccard: {:.4f}, Accuracy: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}'
              .format(epoch+1, args.num_epochs, total_dice/len(domain_loaderT.dataset), jaccard/len(domain_loader_valT.dataset), accuracy/len(domain_loader_valT.dataset),
                     sensitivity/len(domain_loader_valT.dataset), specificity/len(domain_loader_valT.dataset)))
        print(dice_binary/len(domain_loader_valT.dataset), ' ', jaccard_binary/len(domain_loader_valT.dataset))

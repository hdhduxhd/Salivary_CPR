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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
parser.add_argument('--datasetS1', type=str, default='west', help='/kaggle/input/dataset/west')
parser.add_argument('--datasetS2', type=str, default='south', help='/kaggle/input/dataset/south')
parser.add_argument('--datasetT', type=str, default='north', help='/kaggle/input/dataset/north')
parser.add_argument('--batch-size', type=int, default=1, help='batch size for training the model')
parser.add_argument('--data-dir', default='/kaggle/input/dataset', help='data root path')
parser.add_argument('--model_file', default='/kaggle/input/mfsan-trained/west_north2south.pth', help='model path')

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
domain_val1.dataset.transform = composed_transforms_ts
domain_loader_val1 = DataLoader(domain_val1, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

domain_S2 = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetS2, transform=composed_transforms_tr)
train_size = int(train_ratio * len(domain_S2))
test_size = len(domain_S2) - train_size
domain_S2, domain_val2 = torch.utils.data.random_split(domain_S2, [train_size, test_size])
domain_loaderS2 = DataLoader(domain_S2, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
domain_val2.dataset.transform = composed_transforms_ts
domain_loader_val2 = DataLoader(domain_val2, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

domain_T = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetT, transform=composed_transforms_tr)
train_size = int(train_ratio * len(domain_T))
test_size = len(domain_T) - train_size
domain_T, domain_valT = torch.utils.data.random_split(domain_T, [train_size, test_size])
domain_loaderT = DataLoader(domain_T, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
domain_valT.dataset.transform = composed_transforms_ts
domain_loader_valT = DataLoader(domain_valT, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

model = MFSAN(img_size=512).to(device)
model.load_state_dict(torch.load(args.model_file))

#test
mean_dice = 0
model.eval()  # 设置为评估模式
#T
total_dice = 0
jaccard, accuracy, sensitivity, specificity = 0, 0, 0, 0
dice_binary = 0
jaccard_binary = 0
with torch.no_grad():
    for sample in domain_loader_valT:
        inputs = sample['image'].to(device)
        targets = sample['map'].to(device)
        
        pred1, pred2 = model(inputs, mark=3)
        outputs = (pred1 + pred2) / 2
        dice = dice_coeff(outputs, targets)
        total_dice += dice.item()
        jaccard += jaccard_coeff(outputs, targets).item()
        accuracy += pixel_accuracy(outputs, targets).item()
        sensitivity += pixel_sensitivity(outputs, targets).item()
        specificity += pixel_specificity(outputs, targets).item()
      
        outputs = (outputs > 0.5).float()
        dice_binary += dice_coeff(outputs, targets).item()
        jaccard_binary += jaccard_coeff(outputs, targets).item()
print('Target Test Dice: {:.4f}, Jaccard: {:.4f}, Accuracy: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}'
      .format(total_dice/len(domain_loader_valT.dataset), jaccard/len(domain_loader_valT.dataset), accuracy/len(domain_loader_valT.dataset),
              sensitivity/len(domain_loader_valT.dataset), specificity/len(domain_loader_valT.dataset)))
print(dice_binary/len(domain_loader_valT.dataset), ' ', jaccard_binary/len(domain_loader_valT.dataset))

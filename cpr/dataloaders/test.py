import sys
sys.path[0]='/kaggle/working/CPR/cpr'

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr

composed_transforms_train = transforms.Compose([
    tr.Resize1(512, 32),
    tr.add_salt_pepper_noise_sim(),
    tr.adjust_light_sim(),
    tr.eraser_sim(),
    tr.Normalize_tf1(),
    tr.ToTensor1()
])

# 设置随机种子以确保结果可重复
random.seed(42)
db = DL.FundusSegmentation_wsim(base_dir='/kaggle/input/dataset', dataset='south', transform=composed_transforms_train, pseudo='/kaggle/input/checkpoint-best/pseudolabel_south.npz',radius=4)
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

train_loader = DataLoader(db_train, batch_size=8, shuffle=False, num_workers=0)

def test_dataloader():
    return train_loader

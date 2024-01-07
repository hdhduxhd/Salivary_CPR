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

db_train = DL.FundusSegmentation_wsim(base_dir='/kaggle/input/dataset', dataset='south', transform=composed_transforms_train, pseudo='/kaggle/input/checkpoint-best/pseudolabel_south.npz',radius=4)
train_loader = DataLoader(db_train, batch_size=8, shuffle=False, num_workers=0)

def test_dataloader():
    return train_loader

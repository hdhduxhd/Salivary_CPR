import sys
sys.path[0]='/kaggle/working/Salivary_CPR'

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr

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

def test_dataloader(source="west",target="north"):
    domain = DL.FundusSegmentation(base_dir="/kaggle/input/dataset", dataset=source, transform=composed_transforms_tr)
    train_ratio = 0.7
    train_size = int(train_ratio * len(domain))
    test_size = len(domain) - train_size
    domain_S, domain_val = torch.utils.data.random_split(domain, [train_size, test_size])
    domain_loaderS = DataLoader(domain_S, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    domain_val.dataset.transform = composed_transforms_ts
    domain_loader_val = DataLoader(domain_val, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    domain_T = DL.FundusSegmentation(base_dir="/kaggle/input/dataset", dataset=target, transform=composed_transforms_ts)
    domain_loaderT = DataLoader(domain_T, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    return domain_loaderS, domain_loader_val, domain_loaderT

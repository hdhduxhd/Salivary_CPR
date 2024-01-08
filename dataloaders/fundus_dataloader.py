from __future__ import print_function, division
import os
import sys
sys.path[0]='/kaggle/working/CPR'
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from glob import glob
import random

def GetValidTest(base_dir,
                 dataset,
                 split='',
                 valid_ratio=0.25):
    image_list = []
    root_folder = os.path.join(self._base_dir, dataset,split)
    # 遍历根目录下的所有文件夹
    for folder_name in sorted([folder_name for folder_name in os.listdir(root_folder)]):
        folder_path = os.path.join(root_folder, folder_name)
        # 检查是否为文件夹
        if os.path.isdir(folder_path):
            image_folder = os.path.join(folder_path, 'image')
            # 检查image文件夹是否存在
            if os.path.exists(image_folder):
                # 遍历image文件夹下的所有图片文件并按文件名排序
                image_files = sorted(os.listdir(image_folder))
                # 计算中间文件的索引
                middle_index = len(image_files) // 2
                # 取中间的文件
                middle_file = image_files[middle_index]
                image_path = os.path.join(image_folder, middle_file)
                gt_path = image_path.replace('image', 'mask_single')
                self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})
    
    # image_dir = os.path.join(base_dir, dataset, split, 'image')
    # imagelist = glob(image_dir + "/*.png")
    # for image_path in imagelist:
    #     gt_path = image_path.replace('image', 'mask')
    #     image_list.append({'image': image_path, 'label': gt_path, 'id': None})
    
    shuffled_indices = np.random.permutation(len(image_list))
    valid_set_size = int(len(image_list) * valid_ratio)
    valid_indices = shuffled_indices[:valid_set_size]
    test_indices = shuffled_indices[valid_set_size:]
    print('Number of images in {}: {:d}'.format('valid', len(valid_indices)))
    print('Number of images in {}: {:d}'.format('test', len(test_indices)))
    
    return [image_list[i] for i in valid_indices],[image_list[i] for i in test_indices]

class FundusSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir=Path.db_root_dir('fundus'),
                 dataset='west',
                 split='',
                 testid=None,
                 transform=None,
                 image_list=None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        if image_list==None:
            self._base_dir = base_dir
            self.image_list = []
            self.split = split

            self.image_pool = []
            self.label_pool = []
            self.img_name_pool = []

            root_folder = os.path.join(self._base_dir, dataset, self.split)
            # 遍历根目录下的所有文件夹
            for folder_name in sorted([folder_name for folder_name in os.listdir(root_folder)]):
                folder_path = os.path.join(root_folder, folder_name)
                # 检查是否为文件夹
                if os.path.isdir(folder_path):
                    image_folder = os.path.join(folder_path, 'image')
                    # 检查image文件夹是否存在
                    if os.path.exists(image_folder):
                        # 遍历image文件夹下的所有图片文件并按文件名排序
                        image_files = sorted(os.listdir(image_folder))
                        # 计算中间文件的索引
                        middle_index = len(image_files) // 2
                        # 取中间的文件
                        middle_file = image_files[middle_index]
                        image_path = os.path.join(image_folder, middle_file)
                        gt_path = image_path.replace('image', 'mask_single')
                        self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})
            
            # self._image_dir = os.path.join(self._base_dir, dataset, split, 'image')
            # imagelist = glob(self._image_dir + "/*.png")
            # for image_path in imagelist:
            #     gt_path = image_path.replace('image', 'mask')
            #     self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})
            self.transform = transform
            # Display stats
            print('Number of images in {}: {:d}'.format(root_folder, len(self.image_list)))
        else:
            self.image_list = image_list
            self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _target = Image.open(self.image_list[index]['label'])
        if _target.mode == 'RGB':
            _target = _target.convert('L')
        _img_name = self.image_list[index]['image'].split('/')[-3]

        # _img = self.image_pool[index]
        # _target = self.label_pool[index]
        # _img_name = self.img_name_pool[index]
        anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name}

        if self.transform is not None:
            anco_sample = self.transform(anco_sample)

        return anco_sample

    def _read_img_into_memory(self):

        img_num = len(self.image_list)
        for index in range(img_num):
            self.image_pool.append(Image.open(self.image_list[index]['image']).convert('RGB'))
            _target = Image.open(self.image_list[index]['label'])
            if _target.mode == 'RGB':
                _target = _target.convert('L')
            self.label_pool.append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-3]
            self.img_name_pool.append(_img_name)


    def __str__(self):
        return 'Fundus(split=' + str(self.split) + ')'



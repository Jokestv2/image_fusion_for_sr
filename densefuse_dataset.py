# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:03:42 2019

@author: win10
"""

import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

from glob import glob
import os
from PIL import Image
import random


class ImageFusionDataset(Data.Dataset):
    def __init__(self, data_root, transform=None, is_gray=True):
        self.data_root = data_root
        self.is_gray = is_gray
        self.transform = transform

        # use the same seed for transform to get the same cropping area for training
        self.cur_seed = torch.random.seed()

        self.gt_folder_path = os.path.join(data_root, 'GT')
        file_path_list = glob(os.path.join(self.gt_folder_path, '*.*'))

        # it is assumed that corresponding files in all folders have the same name
        self.file_name_list= [os.path.basename(p) for p in file_path_list]
        self.input_folder_path_list = glob(os.path.join(data_root, 'img_*'))

    def __len__(self):
        return len(self.file_name_list)

    def read_img(self, path):
        img = Image.open(path)

        if self.is_gray:
            img = img.convert('L')

        torch.random.manual_seed(self.cur_seed)
        random.seed(self.cur_seed)
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        inputs = {
            f'x{i + 1}': self.read_img(os.path.join(self.input_folder_path_list[i], self.file_name_list[idx]))
            for i in range(len(self.input_folder_path_list))
        }
        inputs['file_name'] = self.file_name_list[idx]

        data_samples = {
            'gt': self.read_img(os.path.join(self.gt_folder_path, self.file_name_list[idx]))
        }
        return inputs, data_samples

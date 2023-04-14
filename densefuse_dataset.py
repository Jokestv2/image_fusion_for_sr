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
    def __init__(self, data_root, mode='train', transform=None, is_gray=True):
        self.data_root = data_root
        self.is_gray = is_gray
        self.transform = transform
        self.mode = mode

        # use the same seed for transform to get the same cropping area for training
        self.cur_seed = torch.random.seed()

        self.gt_folder_path = os.path.join(data_root, 'GT')
        file_path_list = glob(os.path.join(self.gt_folder_path, '*.*'))

        # it is assumed that corresponding files in all folders have the same name
        self.file_name_list = [os.path.basename(p) for p in file_path_list]
        self.input_folder_path_list = glob(os.path.join(data_root, 'img_*'))

    def __len__(self):
        if self.mode == 'train':
            return len(self.file_name_list) * len(self.input_folder_path_list)
        elif self.mode == 'test':
            return len(self.file_name_list)
        else:
            raise RuntimeError(f'mode: {self.mode} is not supported!')

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
        self.cur_seed = torch.random.seed()
        if self.mode == 'train':
            file_i = idx % len(self.file_name_list)
            folder_i = idx // len(self.file_name_list)
            inputs = {
                'x1': self.read_img(os.path.join(self.input_folder_path_list[folder_i], self.file_name_list[file_i]))
            }
        elif self.mode == 'test':
            file_i = idx
            inputs = {
                f'x{i + 1}': self.read_img(os.path.join(self.input_folder_path_list[i], self.file_name_list[file_i]))
                for i in range(len(self.input_folder_path_list))
            }
        else:
            raise RuntimeError(f'mode: {self.mode} is not supported!')

        inputs['file_name'] = self.file_name_list[file_i]
        data_samples = {
            'gt': self.read_img(os.path.join(self.gt_folder_path, self.file_name_list[file_i]))
        }

        return inputs, data_samples

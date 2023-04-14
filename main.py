# -*- coding: utf-8 -*-
"""
CSC2503 final project: image fusion for super-resolution

@author: Ke Zhao
"""
import torch
import argparse
import os

from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from densefuse_net import DenseFusionNet
from test import test, baseline_average
from train import train
from densefuse_dataset import ImageFusionDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description='image fusion model')
    parser.add_argument('mode', help='train or test mode')
    parser.add_argument('--data_root',
                        help='root path of data folder containing train/test/val sub-folders.',
                        default='./data/CUFED5_fusion')
    parser.add_argument('--is_gray', default=False, type=bool)
    parser.add_argument('--checkpoint',
                        help='checkpoint file path to load before training/testing',
                        default='./train_result/model_weight.pkl')
    parser.add_argument('--batch_size',
                        help='batch size for dataloader, only affect training set',
                        default=1,
                        type=int)
    parser.add_argument('--lr', help='learning rate of training', default=1e-5, type=float)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = DenseFusionNet()

    if args.mode == 'train':
        dataset_train = ImageFusionDataset(data_root=os.path.join(args.data_root, 'train'),
                                           mode='train',
                                           transform=transforms.Compose([
                                               transforms.RandomCrop([256, 256], pad_if_needed=True),
                                               transforms.ToTensor(),
                                           ]),
                                           is_gray=args.is_gray)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True)
        dataset_val = ImageFusionDataset(data_root=os.path.join(args.data_root, 'val'),
                                         mode='train',
                                         transform=transforms.ToTensor(),
                                         is_gray=args.is_gray)
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False)

        if len(args.checkpoint) > 0:
            model.load_state_dict(torch.load(args.checkpoint)['weight'])

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train(dataloader_train,
              dataloader_val,
              model,
              optimizer,
              device,
              epochs=10,
              loss_lambda=1,
              val_interval=100)

    elif args.mode == 'test':
        dataset_test = ImageFusionDataset(data_root=os.path.join(args.data_root, 'test'),
                                          mode='test',
                                          transform=transforms.ToTensor(),
                                          is_gray=args.is_gray)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)

        if len(args.checkpoint) > 0:
            model.load_state_dict(torch.load(args.checkpoint)['weight'])

        test(dataloader_test, model, device)

    elif args.mode == 'baseline':
        dataset_test = ImageFusionDataset(data_root=os.path.join(args.data_root, 'test'),
                                          transform=transforms.ToTensor(),
                                          is_gray=args.is_gray)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)

        for fuse_num in range(2, 6):
            baseline_average(dataloader_test, fuse_num=fuse_num)
    else:
        raise RuntimeError(f"mode: {args.mode} not supported!")


if __name__ == '__main__':
    main()
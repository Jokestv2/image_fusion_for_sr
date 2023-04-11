# -*- coding: utf-8 -*-
import os
import torch
import torchvision.transforms as transforms

from utils import mkdir


def test(dataloader, model, device, save_folder_path='./test_result/', fuse_strategy='average', window_width=1):
    mkdir(save_folder_path)
    model = model.to(device)
    for inputs, data_samples in dataloader:
        img1, img2, file_name = inputs['x1'].to(device), inputs['x2'].to(device), inputs['file_name'][0]
        with torch.no_grad():
            if dataloader.dataset.is_gray:
                img = model(img1, img2, fuse_strategy=fuse_strategy)
            else:
                img = torch.cat(
                    [model(img1[:, i, :, :].unsqueeze(1), img2[:, i, :, :].unsqueeze(1))
                     for i in range(3)],
                    dim=1
                )
            # assume that batch_size is 1
            img_fusion = transforms.ToPILImage()(img.squeeze(0).detach().cpu())

            save_path = os.path.join(save_folder_path, file_name)
            img_fusion.save(save_path)

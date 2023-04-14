# -*- coding: utf-8 -*-
import os
import torch
import torchvision.transforms as transforms

from utils import mkdir


def test(dataloader, model, device, fuse_strategy='average', fuse_num=2, window_width=1):
    save_folder_path = os.path.join(dataloader.dataset.data_root, f'fused_cnn_{fuse_num}')
    mkdir(save_folder_path)
    fuse_key_list = [f'x{i}' for i in range(1, fuse_num + 1)]
    model = model.to(device)
    for inputs_raw, data_samples in dataloader:
        file_name = inputs_raw['file_name'][0]
        # there should be at least 1 image to fuse
        assert isinstance(inputs_raw['x1'], torch.Tensor)
        inputs = {}
        for img_key in inputs_raw:
            if img_key in fuse_key_list and isinstance(inputs_raw[img_key], torch.Tensor):
                inputs[img_key] = inputs_raw[img_key].to(device)

        with torch.no_grad():
            if dataloader.dataset.is_gray:
                img = model(inputs, mode='test', fuse_strategy=fuse_strategy)
            else:
                res_list = []
                for i in range(3):
                    inputs_temp = {
                        key: inputs[key][:, i, :, :].unsqueeze(1)
                        for key in inputs
                    }
                    res_list.append(model(inputs_temp, mode='test', fuse_strategy=fuse_strategy))
                img = torch.cat(res_list, dim=1)
            # assume that batch_size is 1
            img_fusion = transforms.ToPILImage()(img.squeeze(0).detach().cpu())

            save_path = os.path.join(save_folder_path, file_name)
            img_fusion.save(save_path)


def baseline_average(dataloader, fuse_num=2):
    save_folder_path = os.path.join(dataloader.dataset.data_root, f'fused_average_{fuse_num}')
    mkdir(save_folder_path)
    fuse_key_list = [f'x{i}' for i in range(1, fuse_num + 1)]
    for inputs, data_samples in dataloader:
        img_avg = 0.
        num_img = 0
        for img_key in inputs:
            if img_key in fuse_key_list and isinstance(inputs[img_key], torch.Tensor):
                img_avg += inputs[img_key][0]
                num_img += 1
        img_avg /= num_img
        img_fusion = transforms.ToPILImage()(img_avg)

        save_path = os.path.join(save_folder_path, inputs['file_name'][0])
        img_fusion.save(save_path)

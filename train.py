# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
from torchvision import transforms
import os

from ssim import SSIM
from utils import mkdir


MSE_fun = nn.MSELoss()
SSIM_fun = SSIM()


def cal_loss(img_fused, img_gt, MSE_fun, SSIM_fun, loss_lambda):
    mse_loss = MSE_fun(img_fused, img_gt)
    ssim_loss = 1 - SSIM_fun(img_fused, img_gt)
    loss = mse_loss + loss_lambda * ssim_loss
    return loss, mse_loss, ssim_loss


def train(dataloader_train,
          dataloader_val,
          model,
          optimizer,
          device,
          save_folder_path='./train_result',
          fuse_strategy='average',
          epochs=10,
          loss_lambda=1,
          val_interval=10):
    mkdir(save_folder_path)
    model = model.to(device)
    for epoch in range(epochs):
        for idx, (inputs, data_samples) in enumerate(dataloader_train):
            img1, img2 = inputs['x1'], inputs['x2']
            img_gt = data_samples['gt']
            if not dataloader_train.dataset.is_gray:
                img1 = img1.reshape(-1, img1.shape[-2], img1.shape[-1]).unsqueeze(1)
                img2 = img2.reshape(-1, img2.shape[-2], img2.shape[-1]).unsqueeze(1)
                img_gt = img_gt.reshape(-1, img_gt.shape[-2], img_gt.shape[-1]).unsqueeze(1)
            img1, img2 = img1.to(device), img2.to(device)
            img_gt = img_gt.to(device)

            optimizer.zero_grad()
            img_fused = model(img1, img2, fuse_strategy=fuse_strategy)
            loss, _, _ = cal_loss(img_fused, img_gt, MSE_fun, SSIM_fun, loss_lambda)
            loss.backward()
            optimizer.step()

            del img1, img2, img_gt
            if idx % val_interval == 0:
                with torch.no_grad():
                    loss_avg, mse_loss_avg, ssim_loss_avg, data_num = 0., 0., 0., 0
                    for inputs_val, data_samples_val in dataloader_val:
                        img1, img2, file_names = inputs_val['x1'], inputs_val['x2'], inputs_val['file_name']
                        img_gt = data_samples_val['gt']
                        if not dataloader_val.dataset.is_gray:
                            img1 = img1.reshape(-1, img1.shape[-2], img1.shape[-1]).unsqueeze(1)
                            img2 = img2.reshape(-1, img2.shape[-2], img2.shape[-1]).unsqueeze(1)
                            img_gt = img_gt.reshape(-1, img_gt.shape[-2], img_gt.shape[-1]).unsqueeze(1)
                        img1, img2 = img1.to(device), img2.to(device)
                        img_gt = img_gt.to(device)

                        img_fused = model(img1, img2, fuse_strategy=fuse_strategy)
                        loss, mse_loss, ssim_loss = cal_loss(img_fused, img_gt, MSE_fun, SSIM_fun, loss_lambda)
                        bs = dataloader_val.batch_size
                        mse_loss_avg += mse_loss.item() * bs
                        ssim_loss_avg += ssim_loss.item() * bs
                        loss_avg += loss.item() * bs
                        data_num += bs

                        # save validation results
                        img_save_folder_path = os.path.join(save_folder_path, 'images', f'epoch_{epoch}_idx_{idx}')
                        mkdir(img_save_folder_path)
                        if not dataloader_val.dataset.is_gray:
                            img_fused = img_fused.reshape(-1, 3, img_fused.shape[-2], img_fused.shape[-1])
                        for img_idx in range(img_fused.shape[0]):
                            img_fusion = transforms.ToPILImage()(img_fused[img_idx].detach().cpu())
                            img_fusion.save(os.path.join(img_save_folder_path, file_names[img_idx]))
                    loss_avg, mse_loss_avg, ssim_loss_avg = loss_avg/data_num, mse_loss_avg/data_num, ssim_loss_avg/data_num
                    print(f"\t Epoch: {epoch}, {idx}/{len(dataloader_train)}, loss_avg: {loss_avg}, mse_loss_avg: {mse_loss_avg}, ssim_loss: {ssim_loss_avg}")

        param_to_save = {
            'weight': model.state_dict,
            'epoch': epoch,
        }
        torch.save(param_to_save, os.path.join(save_folder_path, f'model_weight_epoch{epoch}.pkl'))
        print(f"model saved for epoch: {epoch}")

# -*- coding: utf-8 -*-
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def img_crop():
    src_folder_path = r'E:\CodeProjects\super_resolution\image_fusion_for_sr\data\CUFED5_fusion\test\fused_cnn_2'
    dst_folder_path = r'E:\UofT_MScAC\classes\22Winter\CSC2503 Fundamental of Computer Vision\project\images_to_include'

    img_paths = glob.glob(src_folder_path + '/*.png')
    prefix_list = [39, 10, 80, 93, 65, 91]
    suffix_list = ['cubic', 'hr', 'srcnn', 'esrgan', 'srntt', 'c2', 'c2_hf', 'c2_nn']
    left_top_point = {
        39: (97, 90)
    }
    cropped_w_h = (9, 10)
    for prefix in prefix_list:
        for suffix in suffix_list:
            src_img_path = os.path.join(src_folder_path, f"{prefix}_{suffix}.png")
            img = Image.open(src_img_path)
            file_name = os.path.basename(src_img_path)
            if suffix == 'cubic':
                file_name = f"{prefix}_lr.png"
            left = left_top_point[prefix][0]
            top = left_top_point[prefix][1]
            right = left + cropped_w_h[0]
            bottom = top + cropped_w_h[1]
            img_c = img.crop((left, top, right, bottom))
            img_c.save(os.path.join(dst_folder_path, file_name))


def visual_num_sr_metrics(saved_folder_path, method_name, metrics_dict_list):
    root_path = {}

    ref_num = [1, 2, 3, 4, 5]
    psnr_y_list = []
    ssim_y_list = []
    for rf in ref_num:
        psnr_y_list.append(metrics_dict_list[rf - 1]['psnr_y'])
        ssim_y_list.append(metrics_dict_list[rf - 1]['ssim_y'])

    x_values = ref_num
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    hline_y = [psnr_y_list[0] for x in x_values]
    ln1 = ax1.plot(x_values, psnr_y_list, 'o-b',
                   label=f'PSNR_Y - {method_name}')
    ln2 = ax2.plot(x_values, ssim_y_list, 'o--g',
                   label=f'SSIM - {method_name}')
    ln3 = ax1.plot(x_values, hline_y, 'r--',
                   label=f"PSNR_Y/SSIM - Single Ref {method_name}")

    ax1.set_xticks(x_values, labels=[f'{x:.0f}' for x in x_values])
    ax1.set_xlabel('Number of RefSR Images Fused')

    ax1.set_ylabel('PSNR_Y')
    ax2.set_ylabel('SSIM')

    ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))

    lns = ln3 + ln1 + ln2
    ln_labels = [l.get_label() for l in lns]
    ax1.legend(lns, ln_labels, loc='lower right')

    plt.tight_layout()

    saved_path = os.path.join(saved_folder_path, f'fused_num_{method_name}.png')
    plt.savefig(saved_path, dpi=300, bbox_inches='tight')
    plt.show()

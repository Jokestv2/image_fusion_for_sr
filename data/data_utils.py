"""
Convert results generated by RefSR to a format suitable for fusion model
"""
import os
import shutil
import time
from PIL import Image
import numpy as np

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def mkdirs(dir_path, mode=0o777):
    if dir_path == '':
        return
    if os.path.exists(dir_path):
        new_name = dir_path + '_archived_' + get_time_str()
        print(f'Warning: Path already exists. Rename it to {new_name}', flush=True)
        os.rename(dir_path, new_name)
    dir_paths = os.path.expanduser(dir_path)
    os.makedirs(dir_paths, mode=mode, exist_ok=True)


def mod_crop_and_save(img_path, scale=4):
    """Mod crop images. Applied to ground truth images.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = Image.open(img_path)
    img = np.array(img)
    if img.ndim in [2, 3]:
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    img = Image.fromarray(img)
    img.save(img_path)
    return img

def main():
    result_path = r'.\CUFED5_refsr'
    gt_path = r'.\CUFED5'
    save_path = r'.\CUFED5_fusion'
    num_input = 126
    num_ref = 5
    fusion_subfolders = ['img_' + str(i + 1) for i in range(num_ref)]
    val_idx_list = range(100, 108)
    test_idx_list = [2, 39, 10, 80, 93, 65, 91, 0]
    train_idx_list = list(set(range(num_input)) - set(val_idx_list) - set(test_idx_list))
    fusion_datasets_idx_list = {
        'train': train_idx_list,
        'val': val_idx_list,
        'test': test_idx_list
    }

    for fusion_folder_name in fusion_datasets_idx_list:
        # make root folder for fusion dataset
        fusion_path = os.path.join(save_path, fusion_folder_name)
        mkdirs(fusion_path)

        # Copy ground-truth images
        fusion_gt_path = os.path.join(fusion_path, 'GT')
        os.mkdir(fusion_gt_path)
        for input_idx in fusion_datasets_idx_list[fusion_folder_name]:
            src_file_name = f'{input_idx:03d}_0.png'
            src_file_path = os.path.join(gt_path, src_file_name)
            dst_file_name = f'{input_idx}.png'
            dst_file_path = os.path.join(fusion_gt_path, dst_file_name)
            shutil.copyfile(src_file_path, dst_file_path)
            mod_crop_and_save(dst_file_path)

        # Copy RefSR results
        for (fusion_subfolder, ref_idx) in zip(fusion_subfolders, range(1, 1+num_ref)):
            fusion_subpath = os.path.join(fusion_path, fusion_subfolder)
            os.mkdir(fusion_subpath)
            for input_idx in fusion_datasets_idx_list[fusion_folder_name]:
                src_file_name = f'{input_idx:03d}_{ref_idx}_C2_matching_gan_multi.png'
                src_file_path = os.path.join(result_path, src_file_name)
                dst_file_name = f'{input_idx}.png'
                dst_file_path = os.path.join(fusion_subpath, dst_file_name)
                shutil.copyfile(src_file_path, dst_file_path)
    print("Done!")




if __name__ == '__main__':
    main()
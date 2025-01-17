"""
The function bgr2ycbcr, psnr, _ssim, ssim evaluation code is adapted from openMMCV's implementation so that the
metrics calculation is consistent with the reported values on related papers (e.g. C2-matching, AMSA)
"""
from PIL import Image
import numpy as np
import cv2
import os
from glob import glob
from utils import visual_num_sr_metrics


def evaluate_psnr_ssim_mse_y(img_in_folder, img_gt_folder, is_gray, scale):
    file_path_list = glob(os.path.join(img_gt_folder, '*.*'))
    file_name_list = [os.path.basename(p) for p in file_path_list]

    psnr_y_avg, ssim_y_avg, mse_y_avg = 0., 0., 0.
    for file_name in file_name_list:
        psnr_y, ssim_y, mse_y = comp_psnr_ssim_mse_y(os.path.join(img_in_folder, file_name),
                                                     os.path.join(img_gt_folder, file_name),
                                                     is_gray=is_gray,
                                                     scale=scale)
        psnr_y_avg += psnr_y
        ssim_y_avg += ssim_y
        mse_y_avg += mse_y
    psnr_y_avg /= len(file_name_list)
    ssim_y_avg /= len(file_name_list)
    mse_y_avg /= len(file_name_list)
    return {'psnr_y': psnr_y_avg,
            'ssim_y': ssim_y_avg,
            'mse_y': mse_y_avg}


def comp_psnr_ssim_mse_y(img_in_file, img_gt_file, is_gray, scale):
    if is_gray:
        raise RuntimeError("gray image metric computation not supported yet")
    else:
        img_in_y = bgr2ycbcr(cv2.imread(img_in_file).astype(np.float32) / 255., only_y=True)
        img_gt_y = bgr2ycbcr(cv2.imread(img_gt_file).astype(np.float32) / 255., only_y=True)

    psnr_y = psnr(img_in_y * 255., img_gt_y * 255., crop_border=scale)

    ssim_y = ssim(img_in_y * 255., img_gt_y * 255., crop_border=scale)

    h, w = img_gt_y.shape
    mse_y = np.sum(np.square(img_gt_y - img_in_y)) / (h * w)

    return psnr_y, ssim_y, mse_y


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    img = img.copy()
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img,
                        [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                         [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def psnr(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1, img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1, img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1, img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the SSIM calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


ref_num_list = [1, 2, 3, 4, 5]
metrics_dict_list = []
for i in ref_num_list:
    # pred_path = r'E:\CodeProjects\super_resolution\image_fusion_for_sr\data\CUFED5_fusion\test\fused_cnn_' + str(i)
    pred_path = r'E:\CodeProjects\super_resolution\image_fusion_for_sr\data\CUFED5_fusion\test\fused_average_' + str(i)
    gt_path = r'E:\CodeProjects\super_resolution\image_fusion_for_sr\data\CUFED5_fusion\test\GT'
    res = evaluate_psnr_ssim_mse_y(pred_path, gt_path, False, 4)
    metrics_dict_list.append(res)
    print(res)
saved_path = r'E:\UofT_MScAC\classes\22Winter\CSC2503 Fundamental of Computer Vision\project\images_to_include'
visual_num_sr_metrics(saved_path, 'avg_fusion', metrics_dict_list)
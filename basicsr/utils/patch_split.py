import torch
import numpy as np
import pickle
import cv2

def splitimage(imgtensor, crop_size=128, overlap_size=64):
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts and hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts and wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs : hs + crop_size, ws : ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts


def generate_2d_gaussian_tensor(B, C, H, W, sigma_x, sigma_y):
    # Step 1: 计算中心点
    x_center, y_center = W / 2, H / 2

    # Step 2: 创建网格
    x = torch.arange(W).float() - x_center
    y = torch.arange(H).float() - y_center
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    y, x = torch.meshgrid(y, x, indexing="ij")

    # Step 3: 计算高斯函数
    gaussian = torch.exp(-((x**2) / (2 * sigma_x**2) + (y**2) / (2 * sigma_y**2)))

    # Step 4: 扩展到与输入图像相同的批次大小和通道数
    gaussian = gaussian.expand(B, C, H, W)

    return gaussian


def get_scoremap(H, W, C, B=1, is_x2y2=False, is_gauss=False):  #####默认使用直接平均
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W))
    if is_x2y2 == True:
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (
                    math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-6)
                )
    elif is_gauss == True:
        score = generate_2d_gaussian_tensor(B, C, H, W, int(H / 3.5), int(W / 3.5))
    return score


def mergeimage(
    split_data,
    starts,
    crop_size=128,
    resolution=(1, 3, 128, 128),
    is_x2y2=False,
    is_gauss=False,
):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = torch.zeros((B, C, H, W))
    merge_img = torch.zeros((B, C, H, W))
    scoremap = get_scoremap(
        crop_size, crop_size, C, B=B, is_x2y2=is_x2y2, is_gauss=is_gauss
    )
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs : hs + crop_size, ws : ws + crop_size] += scoremap * simg
        tot_score[:, :, hs : hs + crop_size, ws : ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img

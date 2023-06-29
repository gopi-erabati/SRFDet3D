import cv2
import numpy as np
import os
from tqdm import tqdm
import natsort

# data_root = '/mnt/hdd4/achieve-itn/PhD/Code/workdirs/Results/msf3ddetr_v4/images/'
data_root = '/mnt/hdd4/achieve-itn/PhD/Code/workdirs/Results/msf3ddetr_v5' \
            '-nus/images/'
img_array = []

imgdir_names = os.listdir(data_root)
imgdir_names = natsort.natsorted(imgdir_names)

for imgdir in tqdm(imgdir_names):
    # img_all = np.zeros((450, 1600, 3), dtype=np.uint8)
    img_all = np.zeros((1080, 3840, 3), dtype=np.uint8)

    bev_gt_name = data_root + imgdir + '/bev_gt.png'
    bev_gt = cv2.imread(bev_gt_name)
    bev_gt = cv2.resize(bev_gt, (960, 540), interpolation=cv2.INTER_AREA)

    bev_pred_name = data_root + imgdir + '/bev_pred.png'
    bev_pred = cv2.imread(bev_pred_name)
    bev_pred = cv2.resize(bev_pred, (960, 540), interpolation=cv2.INTER_AREA)

    v1_pred_name = data_root + imgdir + '/v_1_pred.png'
    v1_pred = cv2.imread(v1_pred_name)
    v1_pred = cv2.resize(v1_pred, (960, 540), interpolation=cv2.INTER_AREA)

    v2_pred_name = data_root + imgdir + '/v_2_pred.png'
    v2_pred = cv2.imread(v2_pred_name)
    v2_pred = cv2.resize(v2_pred, (960, 540), interpolation=cv2.INTER_AREA)

    v3_pred_name = data_root + imgdir + '/v_3_pred.png'
    v3_pred = cv2.imread(v3_pred_name)
    v3_pred = cv2.resize(v3_pred, (960, 540), interpolation=cv2.INTER_AREA)

    v4_pred_name = data_root + imgdir + '/v_4_pred.png'
    v4_pred = cv2.imread(v4_pred_name)
    v4_pred = cv2.resize(v4_pred, (960, 540), interpolation=cv2.INTER_AREA)

    v5_pred_name = data_root + imgdir + '/v_5_pred.png'
    v5_pred = cv2.imread(v5_pred_name)
    v5_pred = cv2.resize(v5_pred, (960, 540), interpolation=cv2.INTER_AREA)

    v6_pred_name = data_root + imgdir + '/v_6_pred.png'
    v6_pred = cv2.imread(v6_pred_name)
    v6_pred = cv2.resize(v6_pred, (960, 540), interpolation=cv2.INTER_AREA)

    img_all[0:540, 0:960] = bev_gt
    img_all[540:, 0:960] = bev_pred
    img_all[0:540, 960:1920] = v3_pred
    img_all[0:540, 1920:2880] = v1_pred
    img_all[0:540, 2880:] = v2_pred
    img_all[540:, 960:1920] = v5_pred
    img_all[540:, 1920:2880] = v4_pred
    img_all[540:, 2880:] = v6_pred

    img_array.append(img_all)

out = cv2.VideoWriter('/mnt/hdd4/achieve-itn/PhD/Code/workdirs/Results'
                      '/msf3ddetr_v5-nus/predictions.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), 5, (3840, 1080))

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

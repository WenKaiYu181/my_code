import rasterio
import numpy as np
import os
import cv2

start = 1
final = 26441
for index in range(start,final+1):
    load_path = "L:\\airs\\airs_ratio_cut_1024_2\\x\\" + str(index) + ".tif"
    load_path_mask = "L:\\airs\\airs_ratio_cut_1024_2\\y\\" + str(index) + ".tif"
    load_path_mask_vis = "L:\\airs\\airs_ratio_cut_1024_2\\vis\\" + str(index) + ".tif"

    save_path_x = "L:\\airs\\airs_ratio_cut_1024_resize_256_tif\\x\\" + str(index) + ".tif"
    save_path_y = "L:\\airs\\airs_ratio_cut_1024_resize_256_tif\\y\\" + str(index) + ".tif"
    save_path_image = "L:\\airs\\airs_ratio_cut_1024_resize_256_tif\\vis\\" + str(index) + ".tif"

    img = cv2.imread(load_path)
    mask = cv2.imread(load_path_mask, cv2.IMREAD_GRAYSCALE)
    mask_vis = cv2.imread(load_path_mask_vis, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
    mask_vis = cv2.resize(mask_vis, (256, 256), interpolation=cv2.INTER_AREA)

    cv2.imwrite(save_path_x, img)
    cv2.imwrite(save_path_y, mask)
    cv2.imwrite(save_path_image, mask_vis)
    print("Save File(AIRS)： {}\n".format(index))

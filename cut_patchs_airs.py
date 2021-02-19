import rasterio
import numpy as np
import os
import cv2

count = 0
count_even = 0
skip_counter = 0
area_ratio = 0.1
up = 0.7
TARGET_SIZE = 1024
File_num = 1
image_number = 1
total_num = 857
for i in range(9, 1068):
    filepath = "L:\\airs\\train\\trainval\\train\\image\\" + "christchurch_" + str(i) + ".tif"
    if os.path.isfile(filepath):
        print("檔案存在。")
        print("Start cut {} / {} : {}".format(image_number, total_num, "christchurch_" + str(i) + ".tif"))
        load_path = "L:\\airs\\train\\trainval\\train\\image\\" + "christchurch_" + str(i) + ".tif"
        load_path_mask = "L:\\airs\\train\\trainval\\train\\label\\" + "christchurch_" + str(i) + ".tif"
        load_path_mask_vis = "L:\\airs\\train\\trainval\\train\\label\\" + "christchurch_" + str(i) + "_vis.tif"

        save_path_x = "L:\\airs\\airs_ratio_cut_1024\\x\\"
        save_path_y = "L:\\airs\\airs_ratio_cut_1024\\y\\"
        save_path_image = "L:\\airs\\airs_ratio_cut_1024\\vis\\"

        img = cv2.imread(load_path)
        mask = cv2.imread(load_path_mask, cv2.IMREAD_GRAYSCALE)
        mask_vis = cv2.imread(load_path_mask_vis, cv2.IMREAD_GRAYSCALE)
        if np.max(mask) > 0:
            print("標記存在")
        else:
            print("標記不存在")
            image_number += 1
            continue

        #切割
        for x in range(0, img.shape[1], TARGET_SIZE):
            for y in range(0, img.shape[0] - (TARGET_SIZE - 2)):
                if skip_counter > 0:
                    skip_counter = skip_counter - 1
                    continue

                img_tile = img[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
                mask_tile = mask[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
                mask_vis_tile = mask_vis[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
                if img_tile.shape[0] < TARGET_SIZE or img_tile.shape[1] < TARGET_SIZE:
                    # print("Out image range.\n")
                    continue

                if sum(sum(sum(img_tile)))<=25:
                    continue

                #計算標記比例
                ratio = sum(sum(mask_tile)) / (TARGET_SIZE * TARGET_SIZE)

                if ratio < area_ratio and ratio > 0:
                    continue

                if ratio > up:
                    continue

                if ratio >= area_ratio and ratio <= up:
                    # 確認標記佔全體比例是否超過門檻
                    print("{} / {}  Ratio = {}".format(image_number, total_num, ratio))
                    count = count + 1
                    if count == 2:
                        count_even += 1
                        count = 0
                elif sum(sum(mask_tile)) == 0 and count_even > 0:
                    count_even = count_even - 1
                else:
                    continue

                # if count == 0:
                #   continue

                cv2.imwrite(save_path_x + str(File_num) + ".tif", img_tile)
                cv2.imwrite(save_path_y + str(File_num) + ".tif", mask_tile)
                cv2.imwrite(save_path_image + str(File_num) + ".tif", mask_vis_tile)
                print("{} / {}  Save File(AIRS)： {}\n".format(image_number, total_num, File_num))
                File_num += 1
                skip_counter = TARGET_SIZE - 1

        image_number += 1
    else:
        print("檔案不存在。")

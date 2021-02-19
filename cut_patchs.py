import rasterio
import numpy as np
import os
import cv2

name = ["M-33-7-A-d-2-3.tif",
        "M-33-7-A-d-3-2.tif",
        "M-33-20-D-c-4-2.tif",
        "M-33-20-D-d-3-3.tif",
        "M-33-32-B-b-4-4.tif",
        "M-33-48-A-c-4-4.tif",
        "M-34-5-D-d-4-2.tif",
        "M-34-6-A-d-2-2.tif",
        "M-34-32-B-a-4-3.tif",
        "M-34-32-B-b-1-3.tif",
        "M-34-51-C-b-2-1.tif",
        "M-34-51-C-d-4-1.tif",
        "M-34-55-B-b-4-1.tif",
        "M-34-56-A-b-1-4.tif",
        "M-34-65-D-a-4-4.tif",
        "M-34-65-D-c-4-2.tif",
        "M-34-65-D-d-4-1.tif",
        "M-34-68-B-a-1-3.tif",
        "M-34-77-B-c-2-3.tif",
        "N-33-60-D-c-4-2.tif",
        "N-33-60-D-d-1-2.tif",
        "N-33-96-D-d-1-1.tif",
        "N-33-104-A-c-1-1.tif",
        "N-33-119-C-c-3-3.tif",
        "N-33-130-A-d-3-3.tif",
        "N-33-130-A-d-4-4.tif",
        "N-33-139-C-d-2-2.tif",
        "N-33-139-C-d-2-4.tif",
        "N-33-139-D-c-1-3.tif",
        "N-34-61-B-a-1-1.tif",
        "N-34-66-C-c-4-3.tif",
        "N-34-77-A-b-1-4.tif",
        "N-34-94-A-b-2-4.tif",
        "N-34-97-C-b-1-2.tif",
        "N-34-97-D-c-2-4.tif",
        "N-34-106-A-b-3-4.tif",
        "N-34-106-A-c-1-3.tif",
        "N-34-140-A-b-3-2.tif",
        "N-34-140-A-b-4-2.tif",
        "N-34-140-A-d-3-4.tif",
        "N-34-140-A-d-4-2.tif",]
water = 3
woodland = 2
building = 1

count = 0
count_even = 0
skip_counter = 0
area_ratio = 0.1
up = 0.65
TARGET_SIZE = 256
File_num = 1
for i in range(len(name)):
    print("Start cut {} / {} : {}".format(i + 1, len(name), name[i]))
    load_path = "L:\\landcover.ai\\images\\" + name[i]
    load_path_mask = "L:\\landcover.ai\\masks\\woodland\\" + name[i] + ".png"

    save_path_x = "L:\\landcover.ai\\output\\ratio_cut_woodland\\x\\"
    save_path_y = "L:\\landcover.ai\\output\\ratio_cut_woodland\\y\\"

    save_path_image = "L:\\landcover.ai\\output\\ratio_cut_woodland\\patch_mask\\"

    img = cv2.imread(load_path)
    mask = cv2.imread(load_path_mask, cv2.IMREAD_GRAYSCALE)
    if np.max(mask) > 0:
        mask = mask / np.max(mask)
    else:
        print("No Mask")
        continue

    #切割
    for x in range(0, img.shape[1], TARGET_SIZE):
        for y in range(0, img.shape[0] - (TARGET_SIZE - 2)):
            if skip_counter > 0:
                skip_counter = skip_counter - 1
                continue

            img_tile = img[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
            mask_tile = mask[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
            if img_tile.shape[0] < TARGET_SIZE or img_tile.shape[1] < TARGET_SIZE:
                print("Out image range.\n")
                continue

            #計算比例
            ratio = sum(sum(mask_tile)) / (TARGET_SIZE * TARGET_SIZE)
            if ratio < area_ratio and ratio > 0:
                continue

            if ratio > up:
                continue

            if ratio >= area_ratio and ratio <= up:
                # 確認標記佔全體比例是否超過門檻
                print("{} / {}  Ratio = {}".format(i + 1, len(name), ratio))
                count = count + 1
                if count == 2:
                    count_even += 1
                    count = 0
            elif sum(sum(mask_tile)) == 0 and count_even > 0:
                count_even = count_even - 1
            else:
                continue

            cv2.imwrite(save_path_x + str(File_num) + ".jpg", img_tile)
            cv2.imwrite(save_path_y + str(File_num) + ".png", mask_tile)
            save_mask = (mask_tile * (woodland / 3)) * 255
            cv2.imwrite(save_path_image + str(File_num) + ".png", save_mask)
            print("{} / {}  Save File(woodland)： {}\n".format(i + 1, len(name), File_num))
            File_num += 1
            skip_counter = TARGET_SIZE - 1

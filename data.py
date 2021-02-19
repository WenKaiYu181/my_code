'''
    canny ：21-21*5
    canny input image
        1. 灰階
        2. uint8
        3. threshold ratio
    canny output image
        1. max 255, min 0

    cv2.createTrackbar("滑軌名稱", "視窗名稱", min value, max value, 副函數名稱)

'''
import rasterio
import numpy as np
import os

# import Augmentor
import cv2

# # setup the path of data

load_path = "load path"
# get the data
def load_data(dataset, start_num, total_num, size= (256, 256, 4)):
    print('start load data')
    print('x')
    # variable
    train_x = np.zeros([total_num, size[0], size[1], size[2]], dtype=np.float32)
    train_y = np.zeros([total_num, size[0], size[1], 1], dtype=np.float32)

    # 讀取所有 train x 檔案
    for index in range(start_num, total_num + start_num):
        if ( index - start_num) % 100 == 0:
            print('x data：{}'.format(( index - start_num) / 100))
        # Read file
        data_raster = rasterio.open(load_path + dataset + '\\x\\' + str(index) + '.tif')
        # Read image
        data_raster_img = data_raster.read().transpose((1, 2, 0))
        # data_raster_img = cv2.imread("load_path" + str(index) + '.jpg')
        # Normalize
        train_x[(index - start_num)] = (data_raster_img.astype('float32') / np.max(data_raster_img))

    print('y')
    # 讀取所有 train y 檔案
    for index in range(start_num, total_num+ start_num):
        if( index - start_num) % 100 == 0:
            print('y data：{}'.format(( index - start_num) / 100))
        # Read file
        data_raster = rasterio.open(load_path + dataset + '\\y\\' + str(index) + '.tif')
        # Read mask
        data_raster_nr = data_raster.read(1)
        # Normalize：有值=1,0=0
        for x in range(size[0]):
            for y in range(size[1]):
                if data_raster_nr[x, y] == 0:
                    train_y[ index - start_num, x, y, 0] = 0.0
                else:
                    train_y[ index - start_num, x, y, 0] = 1.0

    print('finish load file.')
    return (train_x, train_y)

#加入額外特徵
def load_data_edge(dataset, start_num, total_num, size= (256, 256, 5)):
    # parameter
    (train_x, train_y) = load_data(dataset, start_num=start_num, total_num=total_num, size=(size[0], size[1],size[2]-1))

    print("Read " + dataset + "gray image.")
    gray_img = np.zeros([total_num, size[0], size[1]], dtype=np.float32)
    for file in range(start_num, start_num + total_num):
        print("Read gray image：" + str(file) + '.png')
        gray = cv2.imread(load_path + dataset + '\\gray\\' + str(file) + '.png', cv2.IMREAD_GRAYSCALE)
        gray_img[file - start_num] = (gray / 255).astype('float32')

    gray_img = np.expand_dims(gray_img, axis=-1)
    train_x = np.concatenate((train_x, gray_img), axis=-1)

    return (train_x, train_y)

#tif轉png
def image_restore(dataset):
    #for file in os.listdir("L:\LAB722\map\data\dataset\dataset_" + dataset + "\\x"):
    for file in os.listdir("load path"):
        # Read file
        print("Read " + file)
        data_raster = rasterio.open('load parh' + file)
        # Read image
        data_raster_img = (data_raster.read().transpose((1, 2, 0)) / np.max(data_raster.read())) * 255
        cv2.imwrite("save path" + file.rstrip(".tif") + '.png', data_raster_img[:,:,0:3].astype('uint8'))

def trans_to_npy(dataset, data_start, data_range):
    (x, y)=load_data_edge(dataset,data_start, data_range, size=(256, 256, 5))
    np.save('save path' + '.npy', x)
    np.save('save path' + '.npy', y)

    (x, y)=load_data(dataset, data_start, data_range, size=(256, 256, 4))
    np.save('save path' + '.npy', x)
    np.save('save path' + '.npy', y)
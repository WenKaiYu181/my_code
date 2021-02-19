from matplotlib import pyplot as plt
from keras.optimizers import *
from keras import backend as K
import tensorflow as tf
#from keras_segmentation.models import *
from keras.models import load_model, Model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from sklearn.ensemble import AdaBoostRegressor

import layers
import model
import model_deeplab3plus
from keras.applications.mobilenet import DepthwiseConv2D
import HRNet
import model_DV3Plus
import loss

import os
import data
import data_preprocessing
import numpy as np
import excel
import estimate
import cv2
import postprocessing
import rasterio

if __name__ == "__main__":
     name = ["save path\\file name"]


# # V2.2
(train_x, train_y) = (np.load(".\\npy\\V2.2_x_1in10230.npy"),
                      np.load(".\\npy\\V2.2_y_1in10230.npy"))
(test_x, test_y) = (np.load(".\\npy\\V2.2_x_10231in2558.npy"),
                    np.load(".\\npy\\V2.2_y_10231in2558.npy"))

# # V2.2_gabor_angle135_psi0_plus_prewitt_x
#(train_x, train_y) = (np.load("L:\\project_code_new\\npy\\V2.2_x_gabor_angle135_psi0_plus_prewitt_x_1in10230.npy"),
#                      np.load("L:\\project_code_new\\npy\\V2.2_y_gabor_angle135_psi0_plus_prewitt_x_1in10230.npy"))
#(test_x, test_y) = (np.load("L:\\project_code_new\\npy\\V2.2_x_gabor_angle135_psi0_plus_prewitt_x_10231_2558.npy"),
#                    np.load("L:\\project_code_new\\npy\\V2.2_y_gabor_angle135_psi0_plus_prewitt_x_10231_2558.npy"))

print("train_x.shape = ",train_x.shape)
print("train_y.shape = ",train_y.shape)
print("test_x.shape = ",test_x.shape)
print("test_y.shape = ",test_y.shape)

weight = 1/7
print("weight = ",weight)

for i in range(len(name)):
    print("Building model.")
    model_select = model.RDB_Branch_Unet((256, 256, 6))

    print("compile model.")
    model_select.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", loss_weights=[weight, weight, weight, weight, weight, weight, weight], metrics=['accuracy'])

    print("model building.")
    model_build = model.model(model=model_select, name=name[0], size=(256, 256, 6))

    print("Select train model：{}".format(model_build.name))

    print("start train.")
    model_build.train(x_train=train_x, y_train=train_y, batch_size=5, epochs=30)

    print("start test.")
    model_build.test(test_x, test_y, data_start=1, batch_size=16)


import numpy as np
from keras.models import *
from keras.layers import *
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
import keras.backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras import backend as keras
import keras
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from matplotlib import pyplot as plt
import excel
import estimate
import cv2
import postprocessing
import numpy as np

# model類別
class model():
    def __init__(self, model, name, size= (256,256,4)):
        self.heights= size[0]
        self.widths= size[1]
        self.channels= size[2]
        self.shape = size
        self.name = name
        self.model = model

    def train(self, x_train, y_train, epochs= 100, batch_size= 5, validation_ratio= 0.125):
        # 紀錄並處存模型權重
        weight_path = ".\\result\\model_record\\" + self.name + '_weight.h5'
        CheckBestPoint = ModelCheckpoint(weight_path + 'bestweights-{epoch:03d}-{val_loss:.5f}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        SaveModel = ModelCheckpoint(weight_path + 'weights-{epoch:03d}-{val_loss:.5f}.h5', verbose=1, save_best_only=False, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
        tsb = TensorBoard(log_dir='save path\\file name')

        if x_train.ndim == 3:
            x_train = np.expand_dims(x_train, axis=-1)
        if y_train.ndim == 3:
            y_train = np.expand_dims(y_train, axis=-1)

        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_ratio, callbacks=[CheckBestPoint, SaveModel, reduce_lr, tsb])
        #history = self.model.fit(x_train, [y_train, y_train, y_train, y_train, y_train, y_train, y_train], batch_size= batch_size, epochs=epochs, validation_split=validation_ratio, callbacks=[CheckBestPoint ,SaveModel ,reduce_lr ,tsb])
        self.model.save(".\\result\model_record\\" + self.name + '.h5')

        print(history.history.keys())
        fig = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower left')
        fig.savefig('.\\result\performance\\' + self.name + '.png')

        ex_loss = excel.Excel()
        ex_loss.write_loss_and_iou(self.name, history.history['loss'], history.history['val_loss'], 0, 0)
        ex_loss.save_excel(file_name=".\\result\data\\" + self.name + "_loss.xlsx")
        ex_loss.close_excel()

    def test(self, x_test, y_test, data_start, batch_size= 10, threshold= 0.5, save_path = None):
        def mkdir(path):
            # 去除首位空格
            path = path.strip()
            # 去除尾部 \ 符號
            path = path.rstrip("\\")

            # 判斷路徑是否存在
            # 存在     True
            # 不存在   False
            isExists = os.path.exists(path)

            # 判斷結果
            if not isExists:
                # 如果不存在則建立目錄
                print("Building the file.")
                # 建立目錄操作函式
                os.makedirs(path)
                return True
            else:
                # 如果目錄存在則不建立，並提示目錄已存在
                print("File is existing.")
                return False

        if x_test.ndim == 3:
            x_test = np.expand_dims(x_test, axis=-1)
        if y_test.ndim == 3:
            y_test = np.expand_dims(y_test, axis=-1)

        y_predict = self.model.predict(x_test, batch_size=batch_size)
        #(y_predict, y_predict_2, y_predict_3, y_predict_4, y_predict_5, y_predict_6, y_predict_7) = self.model.predict(x_test, batch_size=batch_size)

        print("Check the threshold.")
        y_output = postprocessing.check_threshold(y_predict, size=(self.heights, self.widths, 1), threshold= threshold)

        print("Estimate.")
        iou = estimate.IOU(y_test, y_output, self.widths, len(x_test))
        (precision, recall, F1) = estimate.F1_estimate(y_test, y_output, self.widths, len(x_test))
        avr_iou = np.sum(iou) / len(x_test)
        avr_precision = np.sum(precision) / len(x_test)
        avr_recall = np.sum(recall) / len(x_test)
        avr_F1 = np.sum(F1) / len(x_test)
        print("Average IOU：{}".format(avr_iou))

        print('Save the result.')
        mkdir(".\\result\image\\" + self.name)
        for index in range(len(x_test)):
            img_save = y_output[index] * 255
            cv2.imwrite(".\\result\image\\" + self.name + '\\{}.png'.format(data_start + index), img_save)
            print('Save image:{}'.format(data_start + index))

        ex_iou = excel.Excel()
        ex_iou.write_loss_and_iou(self.name, 0, 0, iou, avr_iou)
        #ex_iou.write_loss_and_iou(save_path, 0, 0, iou, avr_iou)
        #ex_iou.write_excel("f4", "precision")
        #ex_iou.write_excel("f5", precision, vertical=True)
        ex_iou.write_excel("f1", "avr_precision", vertical=True)
        ex_iou.write_excel("f2", avr_precision, vertical=True)
        #ex_iou.write_excel("f6", "recall")
        #ex_iou.write_excel("f7", recall, vertical=True)
        ex_iou.write_excel("h1", "avr_recall", vertical=True)
        ex_iou.write_excel("h2", avr_recall, vertical=True)
        #ex_iou.write_excel("f8", "F1")
        #ex_iou.write_excel("f9", F1, vertical=True)
        ex_iou.write_excel("j1", "avr_F1", vertical=True)
        ex_iou.write_excel("j2", avr_F1, vertical=True)

        ex_iou.save_excel(file_name=".\\result\data\\" + self.name + "_iou.xlsx")
        ex_iou.close_excel()


def Unet(size= ( 256, 256, 4)):
    input = Input(size)
    conv1 = Conv2D(64, 3, activation= 'relu', padding= 'same', kernel_initializer='he_normal')(input)
    conv1 = Conv2D(64, 3, activation= 'relu', padding= 'same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D((2,2), None, 'same')(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D((2,2), None, 'same')(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D((2,2), None, 'same')(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D((2,2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    print("U-Net")

    model = Model(input = input, output = conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model


def RDB_Branch_Unet(size= ( 256, 256, 4)):
    def RDBlocks(x, count=3, f=32, o=64): # count參數控制層數
        ## this thing need to be in a damn loop for more customisability
        li = [x]
        pas = Convolution2D(filters=f, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

        for i in range(2, count + 1):
            li.append(pas)
            out = Concatenate()(li)  # conctenated out put
            pas = Convolution2D(filters=f, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)

        # feature extractor from the dense net
        li.append(pas)
        out = Concatenate()(li)
        feat = Convolution2D(filters=o, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(out)

        feat = Add()([feat, x])
        return feat

    def RRDBlocks(x, f=32, o=64):
        RDB = RDBlocks(x, f=f, o=o)
        RDB = RDBlocks(x=RDB, f=f, o=o)
        RDB = RDBlocks(x=RDB, f=f, o=o)

        feat = Add()([RDB, x])
        return feat

    print("RDB_Branch_Unet")
    input = Input(size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    RDB1 = RDBlocks(x=conv1, f=32, o=64)
    pool1 = MaxPooling2D((2, 2), None, 'same')(RDB1)
    conv1_E = Conv2D(32, 1, activation='relu', padding='same', kernel_initializer='he_normal')(RDB1)
    print("conv1_E = ", conv1_E.shape)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    # conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    RDB2 = RDBlocks(x=conv2, f=64, o=128)
    pool2 = MaxPooling2D((2, 2), None, 'same')(RDB2)
    conv2_E = Conv2D(32, 1, activation='relu', padding='same', kernel_initializer='he_normal')(RDB2)
    conv2_E = UpSampling2D(size=(2, 2))(conv2_E)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    # conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    RDB3 = RDBlocks(x=conv3, f=128, o=256)
    pool3 = MaxPooling2D((2, 2), None, 'same')(RDB3)
    conv3_E = Conv2D(32, 1, activation='relu', padding='same', kernel_initializer='he_normal')(RDB3)
    conv3_E = UpSampling2D(size=(2, 2))(conv3_E)
    conv3_E = UpSampling2D(size=(2, 2))(conv3_E)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    # conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    RDB4 = RDBlocks(x=conv4, f=256, o=512)
    drop4 = Dropout(0.5)(RDB4)
    pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)
    conv4_E = Conv2D(32, 1, activation='relu', padding='same', kernel_initializer='he_normal')(RDB4)
    conv4_E = UpSampling2D(size=(2, 2))(conv4_E)
    conv4_E = UpSampling2D(size=(2, 2))(conv4_E)
    conv4_E = UpSampling2D(size=(2, 2))(conv4_E)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    RDB6 = RDBlocks(x=conv6, f=256, o=512)
    conv6_E = Conv2D(32, 1, activation='relu', padding='same', kernel_initializer='he_normal')(RDB6)
    conv6_E = UpSampling2D(size=(2, 2))(conv6_E)
    conv6_E = UpSampling2D(size=(2, 2))(conv6_E)
    conv6_E = UpSampling2D(size=(2, 2))(conv6_E)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(RDB6))
    merge7 = concatenate([RDB3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    # conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    RDB7 = RDBlocks(x=conv7, f=128, o=256)
    conv7_E = Conv2D(32, 1, activation='relu', padding='same', kernel_initializer='he_normal')(RDB7)
    conv7_E = UpSampling2D(size=(2, 2))(conv7_E)
    conv7_E = UpSampling2D(size=(2, 2))(conv7_E)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(RDB7))
    merge8 = concatenate([RDB2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    # conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    RDB8 = RDBlocks(x=conv8, f=64, o=128)
    conv8_E = Conv2D(32, 1, activation='relu', padding='same', kernel_initializer='he_normal')(RDB8)
    conv8_E = UpSampling2D(size=(2, 2))(conv8_E)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(RDB8))
    merge9 = concatenate([RDB1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    # conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    RDB9 = RDBlocks(x=conv9, f=32, o=64)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(RDB9)
    conv10 = Conv2D(1, 1, activation='sigmoid', name="conv10")(conv9)

    concate_E_output_1 = Conv2D(1, 3, activation='sigmoid', padding='same', name="concate_E_output_1")(conv2_E)
    concate_E_output_2 = Conv2D(1, 3, activation='sigmoid', padding='same', name="concate_E_output_2")(conv3_E)
    concate_E_output_3 = Conv2D(1, 3, activation='sigmoid', padding='same', name="concate_E_output_3")(conv4_E)
    concate_E_output_4 = Conv2D(1, 3, activation='sigmoid', padding='same', name="concate_E_output_4")(conv6_E)
    concate_E_output_5 = Conv2D(1, 3, activation='sigmoid', padding='same', name="concate_E_output_5")(conv7_E)
    concate_E_output_6 = Conv2D(1, 3, activation='sigmoid', padding='same', name="concate_E_output_6")(conv8_E)

    model = Model(input=input, output=[conv10, concate_E_output_1, concate_E_output_2, concate_E_output_3, concate_E_output_4, concate_E_output_5, concate_E_output_6])

    return model


def fusion(size= ( 256, 256, 1)):
    print("fusion")
    input_1 = Input(size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

    input_2 = Input(size)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_2)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)

    fusion = concatenate([conv1, conv2])
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(fusion)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

    output = Conv2D(1, 1, activation='sigmoid')(conv3)

    model = Model(input=[input_1, input_2], output=output)

    return model


def ResNet34(input_shape=(256, 256, 4), n_classes=2):
    def Conv2D_BN(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
        if name:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation='relu', name=conv_name)(x)
        x = BatchNormalization(name=bn_name)(x)
        return x

    def identity_block(input_tensor, filters, kernel_size, strides=(1, 1), is_conv_shortcuts=False):
        """
        :param input_tensor:
        :param filters:
        :param kernel_size:
        :param strides:
        :param is_conv_shortcuts: 直接连接或者投影连接
        :return:
        """
        x = Conv2D_BN(input_tensor, filters, kernel_size, strides=strides, padding='same')
        x = Conv2D_BN(x, filters, kernel_size, padding='same')
        if is_conv_shortcuts:
            shortcut = Conv2D_BN(input_tensor, filters, kernel_size, strides=strides, padding='same')
            x = add([x, shortcut])
        else:
            x = add([x, input_tensor])
        return x
    """
    :param input_shape:
    :param n_classes:
    :return:
    """

    input_layer = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(input_layer)
    # block1
    x = Conv2D_BN(x, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    # block2
    x = identity_block(x, filters=64, kernel_size=(3, 3))
    x = identity_block(x, filters=64, kernel_size=(3, 3))
    x = identity_block(x, filters=64, kernel_size=(3, 3))
    # block3
    x = identity_block(x, filters=128, kernel_size=(3, 3), strides=(2, 2), is_conv_shortcuts=True)
    x = identity_block(x, filters=128, kernel_size=(3, 3))
    x = identity_block(x, filters=128, kernel_size=(3, 3))
    x = identity_block(x, filters=128, kernel_size=(3, 3))
    # block4
    x = identity_block(x, filters=256, kernel_size=(3, 3), strides=(2, 2), is_conv_shortcuts=True)
    x = identity_block(x, filters=256, kernel_size=(3, 3))
    x = identity_block(x, filters=256, kernel_size=(3, 3))
    x = identity_block(x, filters=256, kernel_size=(3, 3))
    x = identity_block(x, filters=256, kernel_size=(3, 3))
    x = identity_block(x, filters=256, kernel_size=(3, 3))
    # block5
    x = identity_block(x, filters=512, kernel_size=(3, 3), strides=(2, 2), is_conv_shortcuts=True)
    x = identity_block(x, filters=512, kernel_size=(3, 3))
    x = identity_block(x, filters=512, kernel_size=(3, 3))
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    print("Flatten()(x) : ",x.shape)
    x = Dense(input_shape[0] * input_shape[1], activation='sigmoid')(x)
    print("Dense : ", x.shape)
    x = Reshape((256, 256, 1))(x)
    print("Reshape : ", x.shape)

    model = Model(inputs=input_layer, outputs=x)
    return model


def segnet(input_shape, n_labels, kernel=3, pool_size=(2, 2), output_mode="sigmoid"):
    ###output_mode="softmax"
    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same", kernel_initializer="he_normal")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same", kernel_initializer="he_normal")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Convolution2D(128, (kernel, kernel), padding="same", kernel_initializer="he_normal")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same", kernel_initializer="he_normal")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same", kernel_initializer="he_normal")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same", kernel_initializer="he_normal")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (kernel, kernel), padding="same", kernel_initializer="he_normal")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Convolution2D(512, (kernel, kernel), padding="same", kernel_initializer="he_normal")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(512, (kernel, kernel), padding="same", kernel_initializer="he_normal")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(512, (kernel, kernel), padding="same", kernel_initializer="he_normal")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Convolution2D(512, (kernel, kernel), padding="same", kernel_initializer="he_normal")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (kernel, kernel), padding="same", kernel_initializer="he_normal")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (kernel, kernel), padding="same", kernel_initializer="he_normal")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Convolution2D(512, (kernel, kernel), padding="same", kernel_initializer="he_normal")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(512, (kernel, kernel), padding="same", kernel_initializer="he_normal")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(512, (kernel, kernel), padding="same", kernel_initializer="he_normal")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Convolution2D(512, (kernel, kernel), padding="same", kernel_initializer="he_normal")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(512, (kernel, kernel), padding="same", kernel_initializer="he_normal")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(256, (kernel, kernel), padding="same", kernel_initializer="he_normal")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Convolution2D(256, (kernel, kernel), padding="same", kernel_initializer="he_normal")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(256, (kernel, kernel), padding="same", kernel_initializer="he_normal")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(128, (kernel, kernel), padding="same", kernel_initializer="he_normal")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Convolution2D(128, (kernel, kernel), padding="same", kernel_initializer="he_normal")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (kernel, kernel), padding="same", kernel_initializer="he_normal")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Convolution2D(64, (kernel, kernel), padding="same", kernel_initializer="he_normal")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(n_labels, (1, 1), padding="valid", kernel_initializer="he_normal")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape((input_shape[0], input_shape[1], n_labels))(conv_26)

    outputs = Activation(output_mode)(conv_26)
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")
    # model.compile(optimizer=Adam(lr=1e-4), loss=loss.focal_loss, metrics = ['accuracy'])

    return model





import cv2
import estimate
import numpy as np

# 將結果轉換成1或0
def check_threshold(y_pred, size= (256, 256, 1), threshold= 0.5):
    y_out = np.zeros(((len(y_pred), size[0], size[1], size[2])), dtype= np.uint8)
    print("Check the threshold.")
    for index in range(len(y_pred)):
        for x in range(size[0]):
            for y in range(size[1]):
                if y_pred[index, x, y] > threshold:
                    y_out[index, x, y] = 1
                else:
                    y_out[index, x, y] = 0
        print("threshold image:{}".format(index+1))
    return y_out


import numpy as np

#計算IOU
def IOU(gt, predict, size, number):
    IOU = np.zeros(number)

    for index in range(number):
        area_of_overlap = 0
        area_of_union = 0

        for y in range(size):
            for x in range(size):
                if predict[index, y, x] == 1 and gt[index, y, x] == 1:
                    area_of_overlap += 1

                if predict[index, y, x] == 1 or gt[index, y, x] == 1:
                    area_of_union += 1
        if area_of_overlap == 0 and area_of_union == 0:
            IOU[index] = 1
        else:
            IOU[index] = area_of_overlap / area_of_union
        print("The value of {}：{}\n".format(index+1, IOU[index]))
    return IOU

def F1_estimate(ground_truth, result, size, number):
    precision = np.zeros([number, 1])
    recall = np.zeros([number, 1])
    F1 = np.zeros([number, 1])
    for index in range(number):
        #計算F1-score
        TP = 0
        FP = 0
        FN = 0
        for y in range(size):
            for x in range(size):
                if result[index, y , x ]==1 and ground_truth[index, y , x ]==1 :
                    TP += 1

                if result[index, y , x ]==0 and ground_truth[index, y , x ]==1 :
                    FN += 1

                if result[index, y , x ]==1 and ground_truth[index, y , x ]==0 :
                    FP += 1

                # precision
                if TP==0 and (TP + FP)==0 :
                    precision[index, 0] = 1
                else:
                    precision[index, 0] = TP / (TP + FP)

                # recall
                if TP==0 and (TP + FN)==0 :
                    recall[index, 0] = 1
                else:
                    recall[index, 0] = TP / (TP + FN)

                # F1-score
                if (2 * precision[index, 0] * recall[index, 0]) == 0 and (precision[index, 0] + recall[index, 0]) == 0:
                    F1[index, 0] = 0
                else:
                    F1[index, 0] = (2 * precision[index, 0] * recall[index, 0]) / (precision[index, 0] + recall[index, 0])

    return (precision, recall, F1)
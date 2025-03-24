import os
import random


def Dataset_labels (Rootdata, Training_ratio):
    train_ratio = float(Training_ratio)
    rootdata = Rootdata
    print(train_ratio)
    train_list, test_list = [], []
    data_list = []
    class_flag = -1

    for a, b, c in os.walk(rootdata):
        for i in range(len(c)):
            data_list.append(os.path.join(a, c[i]))
        for i in range(0, int(len(c) * train_ratio)):
            train_data = os.path.join(a, c[i]) + '\t' + str(class_flag) + '\n'
            train_list.append(train_data)
        for i in range(int(len(c) * train_ratio), len(c)):
            test_data = os.path.join(a, c[i]) + '\t' + str(class_flag) + '\n'
            test_list.append(test_data)
        class_flag += 1
    random.shuffle(train_list)
    random.shuffle(test_list)
    with open(rootdata + '/train.txt', 'w', encoding='UTF-8') as F:
        for train_img in train_list:
            F.write(str(train_img))
    with open(rootdata + '/test.txt', 'w', encoding='UTF-8') as F:
        for test_img in test_list:
            F.write(str(test_img))


if __name__ == "__main__":
    RootData = 'C:/Users/Li/Desktop/CNNtest/program_test/dataset_2_corp'
    Training_ratio = 0.8
    Dataset_labels(RootData, Training_ratio)

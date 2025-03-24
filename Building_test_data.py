import os
from pyteomics import mgf
import matplotlib.pyplot as plt
import cv2


def Build_evaluate_figure(input_path, cropped_output_path):
    # for mgf_file in os.listdir(input_path):
    #     file_name = input_path + '/' + mgf_file
    with mgf.read(input_path) as spectra:
        for i, spectrum in enumerate(spectra):
            if spectrum['m/z array'].size > 2:
                a = spectrum['params']['title']
                intensity_max = max(spectrum['intensity array'])  # 归一化
                intensity_min = min(spectrum['intensity array'])  # 归一化
                x = (spectrum['intensity array'] - intensity_min) / (intensity_max - intensity_min)  # 归一化
                plt.figure(figsize=(4, 2), dpi=100)  # 设置图形的长宽都是100个像素，括号内的1乘以dpi100就是图片的像素
                plt.bar(spectrum['m/z array'], x, width=0.05, edgecolor='black')
                plt.xlim(50, 1000)  # 这是设置了横坐标的范围
                image_output_fullname = cropped_output_path + '/' + a + ".png"
                plt.savefig(image_output_fullname)
                plt.close()
                img = cv2.imread(image_output_fullname)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                image1 = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                cropped = image1[30:179, 52:359]  # 裁剪坐标为[y0:y1, x0:x1]
                cropped_image_output_fullname = cropped_output_path + '/' + a + ".png"
                cv2.imwrite(cropped_image_output_fullname, cropped)
                print(i+1)


def CreateEvalData(cropped_output_path, eval_path):
    data_list = []
    #test_root = r"D:/Code_learning/SteroidXtract_codes&manuals_20210201/database/db/MS_figure/new/dataset/val_dataset/"
    for a, b, c in os.walk(cropped_output_path):
        print(c)
        for i in range(len(c)):
            data_list.append(os.path.join(a, c[i]))
    print(data_list)
    with open(eval_path + '/eval.txt',
              'w', encoding='UTF-8') as f:
        for test_img in data_list:
            f.write(test_img + '\t' + "0" + '\n')


def mkdri(input_path):
    folder = os.path.exists(input_path)
    if not folder:
        os.makedirs(input_path)

# def Creat_dataset (rootdata, train_ratio):
#     train_list = []
#     test_list = []
#     data_list = []
#     class_flag = -1
#
#     for a, b, c in os.walk(rootdata):
#         for i in range(len(c)):
#             data_list.append(os.path.join(a, c[i]))
#         for i in range(0, int(len(c) * train_ratio)):
#             train_data = os.path.join(a, c[i]) + '\t' + str(class_flag) + '\n'
#             train_list.append(train_data)
#         for i in range(int(len(c) * train_ratio), len(c)):
#             test_data = os.path.join(a, c[i]) + '\t' + str(class_flag) + '\n'
#             test_list.append(test_data)
#         class_flag += 1
#     print(train_list)
#     random.shuffle(train_list)
#     random.shuffle(test_list)
#     with open(rootdata + '/train.txt', 'w', encoding='UTF-8') as F:
#         for train_img in train_list:
#             F.write(str(train_img))
#     with open(rootdata + '/test.txt', 'w', encoding='UTF-8') as F:
#         for test_img in test_list:
#             F.write(str(test_img))

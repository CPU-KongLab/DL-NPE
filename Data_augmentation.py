from pyteomics import mgf
import matplotlib.pyplot as plt
import heapq
import numpy as np
import cv2
import os

def Data_augmentation(mgf_file, rounds, output_pathway, label):
    file_name = os.path.basename(mgf_file)
    file_name2 = os.path.splitext(file_name)
    for N in range(1, 5):  # 设置噪声水平
        n = N / 10
        for Rounds in range(1, int(rounds) + 1):  # 直接设置扩增几轮
            with mgf.read(mgf_file) as spectra:
                for i, spectrum in enumerate(spectra):
                    if spectrum['m/z array'].size > 10 and spectrum['intensity array'].size > 10:
                        # 设置原始的质谱必须有超过10个离子信号才接受构建质谱图
                        intensity_max = max(spectrum['intensity array'])  # 归一化
                        intensity_min = min(spectrum['intensity array'])  # 归一化
                        x = (spectrum['intensity array'] - intensity_min) / (intensity_max - intensity_min)  # 归一化
                        x_noise = n * np.random.randn(len(x))  # 这句代码是生成一个矩阵，直接和原来的矩阵做运算即可
                        x_data = x + x * x_noise
                        x_data_normalize = (x_data - min(x_data)) / (max(x_data) - min(x_data))
                        plt.figure(figsize=(4, 2), dpi=100)  # 设置图形的长宽都是100个像素，括号内的1乘以dpi100就是图片的像素
                        plt.bar(spectrum['m/z array'], x_data_normalize, width=0.05, edgecolor='black')
                        plt.xlim(50, 1000)  # 这是设置了横坐标的范围
                        plt.savefig(output_pathway + "/" + label + '_' + file_name2[0]
                                    + "_ori_noise_0.%d_augmentation%d_%d.png" % (N, Rounds, i + 1))
                        plt.close()
                        img = cv2.imread(output_pathway + "/" + label + '_' + file_name2[0]
                                         + "_ori_noise_0.%d_augmentation%d_%d.png" % (N, Rounds, i + 1))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        image1 = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                        cropped = image1[30:179, 52:359]  # 裁剪坐标为[y0:y1, x0:x1]
                        cropped_image_output_fullname = output_pathway + "/" + label + '_' + file_name2[0] + \
                                                        "_ori_noise_0.%d_augmentation%d_%d.png" % (N, Rounds, i + 1)
                        cv2.imwrite(cropped_image_output_fullname, cropped)


def Data_augmentation_relative(mgf_file, level, rounds, output_pathway, label):
    file_name = os.path.basename(mgf_file)
    file_name2 = os.path.splitext(file_name)
    with mgf.read(mgf_file) as spectra:
        for i, spectrum in enumerate(spectra):
            progress = (i+1) / len(spectra)
            if spectrum['m/z array'].size > 10 and spectrum['intensity array'].size > 10:
                for limits in range(5, 11):  ###设置的界限
                    limit = limits / int(level)
                    for N in range(1, 5):  ##设置噪声水平
                        n = N / 10
                        for Rounds in range(1, int(rounds) + 1):  ##直接设置扩增几轮
                            ##设置原始的质谱必须有超过10个离子信号才接受构建质谱图
                            # Error = n  ##我现在认为上0.1没问题，明天看一看
                            intensity_max = max(spectrum['intensity array'])  # 归一化
                            intensity_min = min(spectrum['intensity array'])  # 归一化
                            x = (spectrum['intensity array'] - intensity_min) / (intensity_max - intensity_min)  # 归一化
                            x_list = list(x)

                            max_num_index_list = map(x_list.index, heapq.nlargest(int((len(x_list)) * limit), x_list))
                            ##这句代码能够得出原列表中大小在80%以前的数字的序号
                            mass_select = []
                            intensity_select = []
                            for a in list(max_num_index_list):
                                # print(a)
                                mass_select.append(float(spectrum['m/z array'][a]))
                                intensity_select.append(float(x[a]))
                            x_noise = n * np.random.randn(len(intensity_select))  ###这句代码是生成一个矩阵，直接和原来的矩阵做运算即可
                            x_data = intensity_select + intensity_select * x_noise
                            x_data_normalize = (x_data - min(x_data)) / (max(x_data) - min(x_data))
                            plt.figure(figsize=(4, 2), dpi=100)  # 设置图形的长宽都是100个像素，括号内的1乘以dpi100就是图片的像素
                            plt.bar(mass_select, x_data_normalize, width=0.05, edgecolor='black')
                            plt.xlim(50, 1000)  # 这是设置了横坐标的范围
                            plt.savefig(output_pathway + "/" + label + '_' + file_name2[0]
                                        + "_rel_%d_noise_0.%d_augmentation%d_%d.png" % (limits, N, Rounds, i + 1))
                            ### %d就直接用，在括号中排号序就好
                            plt.close()
                            img = cv2.imread(output_pathway + "/" + label + '_' + file_name2[0]
                                             + "_rel_%d_noise_0.%d_augmentation%d_%d.png" % (limits, N, Rounds, i + 1))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            image1 = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                            cropped = image1[30:179, 52:359]  # 裁剪坐标为[y0:y1, x0:x1]
                            cropped_image_output_fullname = output_pathway + "/" + label + '_' + file_name2[0] + \
                                                            "_rel_%d_noise_0.%d_augmentation%d_%d.png" % (
                                                                limits, N, Rounds, i + 1)
                            cv2.imwrite(cropped_image_output_fullname, cropped)
        return progress


def Data_augmentation_absolute (mgf_file, rounds, output_pathway, label):
    file_name = os.path.basename(mgf_file)
    file_name2 = os.path.splitext(file_name)
    for limits in range(1, 5):  ###设置的界限
        limit = limits / 10
        for N in range(1, 5):  ##设置噪声水平
            n = N / 10
            for Rounds in range(1, int(rounds) + 1):  ##直接设置扩增几轮
                with mgf.read(mgf_file) as spectra:
                    for i, spectrum in enumerate(spectra):
                        if spectrum['m/z array'].size > 10 and spectrum['intensity array'].size > 10:
                            intensity_max = max(spectrum['intensity array'])  # 归一化
                            intensity_min = min(spectrum['intensity array'])  # 归一化
                            x = (spectrum['intensity array'] - intensity_min) / (
                                    intensity_max - intensity_min)  # 归一化,这是一个矩阵，要转成列表list才能正常读值
                            x_normalize = []  ##这两个空列表一定要放这里
                            mass_select = []  ##这两个空列表一定要放这里
                            for m, intensity in enumerate(x):
                                if intensity >= limit:
                                    x_normalize.append(intensity)
                                    mass_select.append(spectrum['m/z array'][m])
                            x_noise = n * np.random.randn(len(x_normalize))  ###这句代码是生成一个矩阵，直接和原来的矩阵做运算即可
                            x_data = x_normalize + x_normalize * x_noise  ##数据增强
                            x_data_normalize = (x_data - min(x_data)) / (max(x_data) - min(x_data))
                            plt.figure(figsize=(8, 4), dpi=100)
                            plt.bar(mass_select, x_data_normalize, width=0.05, edgecolor='black')
                            plt.xlim(50, 1000)
                            plt.savefig(output_pathway + "/" + label + '_' + file_name2[0]
                                        + "_abs_%d_noise_0.%d_augmentation%d_%d.png" % (limits, N, Rounds, i + 1))
                            plt.close()
                            img = cv2.imread(output_pathway + "/" + label + '_' + file_name2[0]
                                             + "_abs_%d_noise_0.%d_augmentation%d_%d.png" % (limits, N, Rounds, i + 1))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            image1 = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                            cropped = image1[58:357, 103:718]  # 裁剪坐标为[y0:y1, x0:x1]
                            cropped_image_output_fullname = output_pathway + "/" + label + '_' + file_name2[0] + \
                                                            "_abs_%d_noise_0.%d_augmentation%d_%d.png" % \
                                                            (limits, N, Rounds, i + 1)
                            cv2.imwrite(cropped_image_output_fullname, cropped)


if __name__ == "__main__":
    mgf_file = 'C:/Users/Li/Desktop/CNNtest/test/CTB.mgf'
    level = 100
    rounds = 3
    output_pathway = 'C:/Users/Li/Desktop/CNNtest/test'
    label = 'mono'
    Data_augmentation_relative(mgf_file, level, rounds, output_pathway, label)
    print(Data_augmentation_relative(mgf_file, level, rounds, output_pathway, label))

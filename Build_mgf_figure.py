from pyteomics import mgf
import json
import matplotlib.pyplot as plt
import time
import cv2
import os


def Build_MS_Figure(input_path, output_path, cropped_output_path):
    file_name = os.path.basename(input_path)
    file_name2 = os.path.splitext(file_name)
    if '.mgf' in input_path:
        with mgf.read(input_path) as spectra:
            for i, spectrum in enumerate(spectra):
                # progress = (i + 1) / len(spectra)
                if spectrum['m/z array'].size > 2:
                    # a = spectrum['params']['title']
                    intensity_max = max(spectrum['intensity array'])  # 归一化
                    intensity_min = min(spectrum['intensity array'])  # 归一化
                    x = (spectrum['intensity array'] - intensity_min) / (intensity_max - intensity_min)  # 归一化
                    plt.figure(figsize=(8, 4), dpi=100)  # 设置图形的长宽都是100个像素，括号内的1乘以dpi100就是图片的像素
                    plt.bar(spectrum['m/z array'], x, width=0.05, edgecolor='black')
                    plt.xlim(50, 1000)  # 这是设置了横坐标的范围
                    image_output_fullname = output_path + '/' + file_name2[0] + '_' + str(i) + "_ori.png"
                    plt.savefig(image_output_fullname)
                    plt.close()
                    img = cv2.imread(image_output_fullname)
                    # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    image1 = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    cropped = image1[58:357, 103:718]  # 裁剪坐标为[y0:y1, x0:x1] 原来是：30:179, 52:359
                    cropped_image_output_fullname = cropped_output_path + '/' + file_name2[0] + '_' + str(i) \
                                                    + "_ori.png"
                    cv2.imwrite(cropped_image_output_fullname, cropped)
        # return progress

    if '.json' in input_path:
        with open(input_path, errors='ignore') as f:
            spectra = json.load(f)
            for i, item in enumerate(spectra):
                a = item['spectrum']
                b = a.split(' ')
                mass_array = []  # 一定要把空列表建在循环外面，然后经过下面的for循环才能一个一个把列表填满
                intensity_array = []
                for x in range(0, len(b)):
                    c = (b[x]).split(':')
                    mass = float(c[0])
                    intensity = float(c[1]) / 100
                    mass_array.append(mass)
                    intensity_array.append(intensity)
                if len(mass_array) > 10 and len(intensity_array) > 10:
                    plt.figure(figsize=(8, 4), dpi=100)  # 设置图形的长宽都是100个像素，括号内的1乘以dpi100就是图片的像素
                    plt.bar(mass_array, intensity_array, width=0.05, edgecolor='black')
                    plt.xlim(50, 1000)  # 这是设置了横坐标的范围
                    image_output_fullname = output_path + '/' + file_name2[0] + '_' + "json_%d.png" % (i + 1)
                    plt.savefig(image_output_fullname)
                    plt.close()
                    img = cv2.imread(image_output_fullname)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    image1 = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    cropped = image1[58:357, 103:718]  # 裁剪坐标为[y0:y1, x0:x1]
                    cropped_image_output_fullname = cropped_output_path + '/' + file_name2[0] + '_' \
                                                    + "json_%d.png" % (i + 1)
                    cv2.imwrite(cropped_image_output_fullname, cropped)

import heapq
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from Build_mgf_figure import Build_MS_Figure
from Data_augmentation import Data_augmentation, Data_augmentation_relative, Data_augmentation_absolute
from multiprocessing import Process
import multiprocessing
import sys
from Dataset_labels import Dataset_labels
import Training_model
from CreateDataloader import LoadData
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch import save
from torch import cuda
from torch import nn
from torch import load
from Characteristic_ion_search import CI_search
from Building_test_data import Build_evaluate_figure, CreateEvalData, mkdri
import os
from pyteomics import mgf
import pandas as pd
from Testing import eval
import csv
import ctypes
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

    def setupUi(self, MainWindow):
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("C:/Deeplearning/Deep learning/Deep-NPExtractor/DL-NPE.png"),
                       QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(471, 629)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 471, 591))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.pushButton = QtWidgets.QPushButton(self.tab)
        self.pushButton.setGeometry(QtCore.QRect(20, 50, 75, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.tab)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 120, 75, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_2.setGeometry(QtCore.QRect(130, 120, 191, 21))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setGeometry(QtCore.QRect(180, 30, 51, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.lineEdit = QtWidgets.QLineEdit(self.tab)
        self.lineEdit.setGeometry(QtCore.QRect(130, 50, 151, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setGeometry(QtCore.QRect(170, 100, 51, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_4.setGeometry(QtCore.QRect(130, 170, 91, 20))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.label_8 = QtWidgets.QLabel(self.tab)
        self.label_8.setGeometry(QtCore.QRect(150, 150, 51, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.tab)
        self.label_9.setGeometry(QtCore.QRect(400, 220, 41, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.pushButton_5 = QtWidgets.QPushButton(self.tab)
        self.pushButton_5.setGeometry(QtCore.QRect(20, 240, 75, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.label_10 = QtWidgets.QLabel(self.tab)
        self.label_10.setGeometry(QtCore.QRect(180, 220, 61, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.tab)
        self.label_11.setGeometry(QtCore.QRect(150, 270, 51, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.lineEdit_10 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_10.setGeometry(QtCore.QRect(130, 290, 91, 20))
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.lineEdit_13 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_13.setGeometry(QtCore.QRect(130, 410, 91, 20))
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.pushButton_6 = QtWidgets.QPushButton(self.tab)
        self.pushButton_6.setGeometry(QtCore.QRect(20, 360, 75, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setObjectName("pushButton_6")
        self.label_14 = QtWidgets.QLabel(self.tab)
        self.label_14.setGeometry(QtCore.QRect(150, 390, 51, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.tab)
        self.label_15.setGeometry(QtCore.QRect(150, 460, 151, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.pushButton_7 = QtWidgets.QPushButton(self.tab)
        self.pushButton_7.setGeometry(QtCore.QRect(20, 480, 75, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.pushButton_7.setFont(font)
        self.pushButton_7.setObjectName("pushButton_7")
        self.lineEdit_15 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_15.setGeometry(QtCore.QRect(120, 480, 161, 20))
        self.lineEdit_15.setObjectName("lineEdit_15")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_5.setGeometry(QtCore.QRect(300, 50, 151, 20))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.label_23 = QtWidgets.QLabel(self.tab)
        self.label_23.setGeometry(QtCore.QRect(330, 30, 101, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(self.tab)
        self.label_24.setGeometry(QtCore.QRect(310, 150, 101, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_24.setFont(font)
        self.label_24.setObjectName("label_24")
        self.lineEdit_17 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_17.setGeometry(QtCore.QRect(250, 170, 201, 20))
        self.lineEdit_17.setObjectName("lineEdit_17")
        self.label_13 = QtWidgets.QLabel(self.tab)
        self.label_13.setGeometry(QtCore.QRect(200, 340, 61, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.lineEdit_18 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_18.setGeometry(QtCore.QRect(250, 290, 201, 20))
        self.lineEdit_18.setObjectName("lineEdit_18")
        self.label_25 = QtWidgets.QLabel(self.tab)
        self.label_25.setGeometry(QtCore.QRect(310, 270, 101, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_25.setFont(font)
        self.label_25.setObjectName("label_25")
        self.lineEdit_19 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_19.setGeometry(QtCore.QRect(250, 410, 201, 20))
        self.lineEdit_19.setObjectName("lineEdit_19")
        self.label_26 = QtWidgets.QLabel(self.tab)
        self.label_26.setGeometry(QtCore.QRect(310, 390, 101, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_26.setFont(font)
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(self.tab)
        self.label_27.setGeometry(QtCore.QRect(380, 100, 41, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_27.setFont(font)
        self.label_27.setObjectName("label_27")
        self.lineEdit_9 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_9.setGeometry(QtCore.QRect(130, 240, 141, 21))
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.lineEdit_11 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_11.setGeometry(QtCore.QRect(380, 240, 71, 20))
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.lineEdit_21 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_21.setGeometry(QtCore.QRect(290, 240, 71, 20))
        self.lineEdit_21.setObjectName("lineEdit_21")
        self.label_28 = QtWidgets.QLabel(self.tab)
        self.label_28.setGeometry(QtCore.QRect(310, 220, 41, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_28.setFont(font)
        self.label_28.setObjectName("label_28")
        self.lineEdit_12 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_12.setGeometry(QtCore.QRect(130, 360, 191, 21))
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.lineEdit_22 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_22.setGeometry(QtCore.QRect(340, 360, 111, 20))
        self.lineEdit_22.setObjectName("lineEdit_22")
        self.label_29 = QtWidgets.QLabel(self.tab)
        self.label_29.setGeometry(QtCore.QRect(380, 340, 41, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_29.setFont(font)
        self.label_29.setObjectName("label_29")
        self.lineEdit_23 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_23.setGeometry(QtCore.QRect(340, 120, 111, 20))
        self.lineEdit_23.setObjectName("lineEdit_23")
        self.lineEdit_20 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_20.setGeometry(QtCore.QRect(290, 480, 161, 20))
        self.lineEdit_20.setObjectName("lineEdit_20")
        self.label_30 = QtWidgets.QLabel(self.tab)
        self.label_30.setGeometry(QtCore.QRect(320, 460, 101, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_30.setFont(font)
        self.label_30.setObjectName("label_30")
        self.groupBox = QtWidgets.QGroupBox(self.tab)
        self.groupBox.setGeometry(QtCore.QRect(10, 20, 451, 61))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 89, 451, 111))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_3.setGeometry(QtCore.QRect(9, 210, 451, 111))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.groupBox_4 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_4.setGeometry(QtCore.QRect(10, 330, 451, 111))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.groupBox_5 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_5.setGeometry(QtCore.QRect(10, 449, 451, 61))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")
        self.groupBox_5.raise_()
        self.groupBox_4.raise_()
        self.groupBox_3.raise_()
        self.groupBox_2.raise_()
        self.groupBox.raise_()
        self.pushButton.raise_()
        self.pushButton_2.raise_()
        self.lineEdit_2.raise_()
        self.label_2.raise_()
        self.lineEdit.raise_()
        self.label_3.raise_()
        self.lineEdit_4.raise_()
        self.label_8.raise_()
        self.label_9.raise_()
        self.pushButton_5.raise_()
        self.label_10.raise_()
        self.label_11.raise_()
        self.lineEdit_10.raise_()
        self.lineEdit_13.raise_()
        self.pushButton_6.raise_()
        self.label_14.raise_()
        self.label_15.raise_()
        self.pushButton_7.raise_()
        self.lineEdit_15.raise_()
        self.lineEdit_5.raise_()
        self.label_23.raise_()
        self.label_24.raise_()
        self.lineEdit_17.raise_()
        self.label_13.raise_()
        self.lineEdit_18.raise_()
        self.label_25.raise_()
        self.lineEdit_19.raise_()
        self.label_26.raise_()
        self.label_27.raise_()
        self.lineEdit_9.raise_()
        self.lineEdit_11.raise_()
        self.lineEdit_21.raise_()
        self.label_28.raise_()
        self.lineEdit_12.raise_()
        self.lineEdit_22.raise_()
        self.label_29.raise_()
        self.lineEdit_23.raise_()
        self.lineEdit_20.raise_()
        self.label_30.raise_()
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_6.setGeometry(QtCore.QRect(140, 50, 121, 21))
        self.lineEdit_6.setObjectName("lineEdit_6")

        items = ['NeuralNetwork1', 'NeuralNetwork2', 'NeuralNetwork3']  # 新加
        self.comboBox = QtWidgets.QComboBox(self.tab_2)
        self.comboBox.addItems(items)
        self.comboBox.setCurrentIndex(0)  # 新加，设置默认值
        self.comboBox.currentText()  # 新加，获得当前内容
        self.comboBox.setGeometry(QtCore.QRect(140, 20, 121, 22))
        self.comboBox.setObjectName("comboBox")
        self.textBrowser = QtWidgets.QTextBrowser(self.tab_2)
        self.textBrowser.setGeometry(QtCore.QRect(20, 360, 421, 192))
        self.textBrowser.setObjectName("textBrowser")
        self.lineEdit_25 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_25.setGeometry(QtCore.QRect(140, 80, 121, 21))
        self.lineEdit_25.setObjectName("lineEdit_25")
        self.progressBar = QtWidgets.QProgressBar(self.tab_2)
        self.progressBar.setGeometry(QtCore.QRect(130, 310, 291, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setGeometry(QtCore.QRect(10, 110, 151, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.tab_2)
        self.label_6.setGeometry(QtCore.QRect(10, 140, 151, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.lineEdit_26 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_26.setGeometry(QtCore.QRect(140, 110, 301, 21))
        self.lineEdit_26.setObjectName("lineEdit_26")
        self.lineEdit_27 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_27.setGeometry(QtCore.QRect(140, 140, 301, 21))
        self.lineEdit_27.setObjectName("lineEdit_27")
        self.label_7 = QtWidgets.QLabel(self.tab_2)
        self.label_7.setGeometry(QtCore.QRect(10, 170, 151, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.lineEdit_28 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_28.setGeometry(QtCore.QRect(140, 170, 101, 21))
        self.lineEdit_28.setObjectName("lineEdit_28")
        self.label_16 = QtWidgets.QLabel(self.tab_2)
        self.label_16.setGeometry(QtCore.QRect(10, 230, 151, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.lineEdit_29 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_29.setGeometry(QtCore.QRect(140, 230, 301, 21))
        self.lineEdit_29.setObjectName("lineEdit_29")
        self.label_17 = QtWidgets.QLabel(self.tab_2)
        self.label_17.setGeometry(QtCore.QRect(20, 310, 151, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.tab_2)
        self.label_18.setGeometry(QtCore.QRect(200, 340, 151, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.label_12 = QtWidgets.QLabel(self.tab_2)
        self.label_12.setGeometry(QtCore.QRect(10, 20, 151, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.label_31 = QtWidgets.QLabel(self.tab_2)
        self.label_31.setGeometry(QtCore.QRect(10, 50, 151, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_31.setFont(font)
        self.label_31.setObjectName("label_31")
        self.label_32 = QtWidgets.QLabel(self.tab_2)
        self.label_32.setGeometry(QtCore.QRect(10, 80, 151, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_32.setFont(font)
        self.label_32.setObjectName("label_32")
        self.pushButton_3 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 270, 431, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.lineEdit_30 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_30.setGeometry(QtCore.QRect(140, 200, 101, 21))
        self.lineEdit_30.setObjectName("lineEdit_30")
        self.label_37 = QtWidgets.QLabel(self.tab_2)
        self.label_37.setGeometry(QtCore.QRect(10, 200, 151, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_37.setFont(font)
        self.label_37.setObjectName("label_37")
        self.groupBox_9 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_9.setGeometry(QtCore.QRect(0, 0, 461, 261))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.groupBox_9.setFont(font)
        self.groupBox_9.setObjectName("groupBox_9")
        self.groupBox_9.raise_()
        self.lineEdit_6.raise_()
        self.comboBox.raise_()
        self.textBrowser.raise_()
        self.lineEdit_25.raise_()
        self.progressBar.raise_()
        self.label_5.raise_()
        self.label_6.raise_()
        self.lineEdit_26.raise_()
        self.lineEdit_27.raise_()
        self.label_7.raise_()
        self.lineEdit_28.raise_()
        self.label_16.raise_()
        self.lineEdit_29.raise_()
        self.label_17.raise_()
        self.label_18.raise_()
        self.label_12.raise_()
        self.label_31.raise_()
        self.label_32.raise_()
        self.pushButton_3.raise_()
        self.lineEdit_30.raise_()
        self.label_37.raise_()
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.lineEdit_40 = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_40.setGeometry(QtCore.QRect(310, 250, 131, 21))
        self.lineEdit_40.setObjectName("lineEdit_40")
        self.pushButton_23 = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_23.setGeometry(QtCore.QRect(150, 220, 161, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.pushButton_23.setFont(font)
        self.pushButton_23.setObjectName("pushButton_23")
        self.pushButton_24 = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_24.setGeometry(QtCore.QRect(20, 140, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.pushButton_24.setFont(font)
        self.pushButton_24.setObjectName("pushButton_24")
        self.label_21 = QtWidgets.QLabel(self.tab_3)
        self.label_21.setGeometry(QtCore.QRect(280, 390, 141, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.comboBox_2 = QtWidgets.QComboBox(self.tab_3)
        self.comboBox_2.addItems(items)
        self.comboBox_2.setCurrentIndex(0)  # 新加，设置默认值
        self.comboBox_2.currentText()  # 新加，获得当前内容
        self.comboBox_2.setGeometry(QtCore.QRect(96, 250, 120, 21))
        self.comboBox_2.setObjectName("comboBox_2")
        self.label_22 = QtWidgets.QLabel(self.tab_3)
        self.label_22.setGeometry(QtCore.QRect(220, 250, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.label_44 = QtWidgets.QLabel(self.tab_3)
        self.label_44.setGeometry(QtCore.QRect(80, 290, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_44.setFont(font)
        self.label_44.setObjectName("label_44")
        self.lineEdit_45 = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_45.setGeometry(QtCore.QRect(20, 310, 201, 21))
        self.lineEdit_45.setObjectName("lineEdit_45")
        self.label_45 = QtWidgets.QLabel(self.tab_3)
        self.label_45.setGeometry(QtCore.QRect(310, 290, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_45.setFont(font)
        self.label_45.setObjectName("label_45")
        self.lineEdit_46 = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_46.setGeometry(QtCore.QRect(240, 310, 201, 21))
        self.lineEdit_46.setObjectName("lineEdit_46")
        self.label_46 = QtWidgets.QLabel(self.tab_3)
        self.label_46.setGeometry(QtCore.QRect(90, 340, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_46.setFont(font)
        self.label_46.setObjectName("label_46")
        self.lineEdit_47 = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_47.setGeometry(QtCore.QRect(20, 360, 201, 21))
        self.lineEdit_47.setObjectName("lineEdit_47")
        self.label_47 = QtWidgets.QLabel(self.tab_3)
        self.label_47.setGeometry(QtCore.QRect(300, 340, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_47.setFont(font)
        self.label_47.setObjectName("label_47")
        self.lineEdit_73 = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_73.setGeometry(QtCore.QRect(240, 360, 201, 21))
        self.lineEdit_73.setObjectName("lineEdit_73")
        self.label_73 = QtWidgets.QLabel(self.tab_3)
        self.label_73.setGeometry(QtCore.QRect(70, 390, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_73.setFont(font)
        self.label_73.setObjectName("label_73")
        self.lineEdit_74 = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_74.setGeometry(QtCore.QRect(20, 410, 201, 21))
        self.lineEdit_74.setObjectName("lineEdit_74")
        self.lineEdit_75 = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_75.setGeometry(QtCore.QRect(240, 410, 201, 21))
        self.lineEdit_75.setObjectName("lineEdit_75")
        self.lineEdit_14 = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_14.setGeometry(QtCore.QRect(290, 50, 151, 21))
        self.lineEdit_14.setObjectName("lineEdit_14")
        self.label_34 = QtWidgets.QLabel(self.tab_3)
        self.label_34.setGeometry(QtCore.QRect(320, 30, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_34.setFont(font)
        self.label_34.setObjectName("label_34")
        self.lineEdit_76 = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_76.setGeometry(QtCore.QRect(20, 460, 201, 21))
        self.lineEdit_76.setObjectName("lineEdit_76")
        self.groupBox_6 = QtWidgets.QGroupBox(self.tab_3)
        self.groupBox_6.setGeometry(QtCore.QRect(10, 10, 441, 71))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.groupBox_6.setFont(font)
        self.groupBox_6.setObjectName("groupBox_6")
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_6)
        self.pushButton_8.setGeometry(QtCore.QRect(10, 40, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.pushButton_8.setFont(font)
        self.pushButton_8.setObjectName("pushButton_8")
        self.lineEdit_7 = QtWidgets.QLineEdit(self.groupBox_6)
        self.lineEdit_7.setGeometry(QtCore.QRect(110, 40, 151, 21))
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.label_33 = QtWidgets.QLabel(self.groupBox_6)
        self.label_33.setGeometry(QtCore.QRect(140, 20, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_33.setFont(font)
        self.label_33.setObjectName("label_33")
        self.groupBox_7 = QtWidgets.QGroupBox(self.tab_3)
        self.groupBox_7.setGeometry(QtCore.QRect(10, 100, 441, 71))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.groupBox_7.setFont(font)
        self.groupBox_7.setObjectName("groupBox_7")
        self.lineEdit_8 = QtWidgets.QLineEdit(self.groupBox_7)
        self.lineEdit_8.setGeometry(QtCore.QRect(110, 40, 151, 21))
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.label_19 = QtWidgets.QLabel(self.groupBox_7)
        self.label_19.setGeometry(QtCore.QRect(140, 20, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.groupBox_7)
        self.label_20.setGeometry(QtCore.QRect(310, 20, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.lineEdit_16 = QtWidgets.QLineEdit(self.groupBox_7)
        self.lineEdit_16.setGeometry(QtCore.QRect(280, 40, 151, 21))
        self.lineEdit_16.setObjectName("lineEdit_16")
        self.groupBox_8 = QtWidgets.QGroupBox(self.tab_3)
        self.groupBox_8.setGeometry(QtCore.QRect(10, 200, 441, 291))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.groupBox_8.setFont(font)
        self.groupBox_8.setObjectName("groupBox_8")
        self.label_35 = QtWidgets.QLabel(self.groupBox_8)
        self.label_35.setGeometry(QtCore.QRect(10, 50, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_35.setFont(font)
        self.label_35.setObjectName("label_35")
        self.label_74 = QtWidgets.QLabel(self.groupBox_8)
        self.label_74.setGeometry(QtCore.QRect(60, 240, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_74.setFont(font)
        self.label_74.setObjectName("label_74")
        self.label_36 = QtWidgets.QLabel(self.groupBox_8)
        self.label_36.setGeometry(QtCore.QRect(290, 240, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_36.setFont(font)
        self.label_36.setObjectName("label_36")
        self.lineEdit_77 = QtWidgets.QLineEdit(self.groupBox_8)
        self.lineEdit_77.setGeometry(QtCore.QRect(230, 260, 201, 21))
        self.lineEdit_77.setObjectName("lineEdit_77")
        self.groupBox_8.raise_()
        self.groupBox_7.raise_()
        self.groupBox_6.raise_()
        self.lineEdit_40.raise_()
        self.pushButton_23.raise_()
        self.pushButton_24.raise_()
        self.label_21.raise_()
        self.comboBox_2.raise_()
        self.label_22.raise_()
        self.label_44.raise_()
        self.lineEdit_45.raise_()
        self.label_45.raise_()
        self.lineEdit_46.raise_()
        self.label_46.raise_()
        self.lineEdit_47.raise_()
        self.label_47.raise_()
        self.lineEdit_73.raise_()
        self.label_73.raise_()
        self.lineEdit_74.raise_()
        self.lineEdit_75.raise_()
        self.lineEdit_14.raise_()
        self.label_34.raise_()
        self.lineEdit_76.raise_()
        self.tabWidget.addTab(self.tab_3, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 471, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actioninput = QtWidgets.QAction(MainWindow)
        self.actioninput.setObjectName("actioninput")

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(lambda: self.build_mgf())  # 加上的
        self.pushButton_2.clicked.connect(lambda: self.Data_augmentation())  # 加上的
        self.pushButton_5.clicked.connect(lambda: self.Data_augmentation_relative())  # 加上的
        self.pushButton_6.clicked.connect(lambda: self.Data_augmentation_absolute())  # 加上的
        self.pushButton_3.clicked.connect(lambda: self.Training_model())  # 加上的
        self.pushButton_7.clicked.connect(lambda: self.Build_label())  # 加上的
        self.pushButton_8.clicked.connect(lambda: self.searching_characteristic_ion())  # 加上的
        self.pushButton_24.clicked.connect(lambda: self.Building_evaluation_data())  # 加上的
        self.pushButton_23.clicked.connect(lambda: self.Testing())  # 加上的

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DL-NPE"))
        self.pushButton.setText(_translate("MainWindow", "Run"))
        self.pushButton_2.setText(_translate("MainWindow", "Run"))
        self.label_2.setText(_translate("MainWindow", "mgf file"))
        self.label_3.setText(_translate("MainWindow", "mgf file"))
        self.label_8.setText(_translate("MainWindow", "Rounds"))
        self.label_9.setText(_translate("MainWindow", "Level"))
        self.pushButton_5.setText(_translate("MainWindow", "Run"))
        self.label_10.setText(_translate("MainWindow", "mgf file"))
        self.label_11.setText(_translate("MainWindow", "Rounds"))
        self.pushButton_6.setText(_translate("MainWindow", "Run"))
        self.label_14.setText(_translate("MainWindow", "Rounds"))
        self.label_15.setText(_translate("MainWindow", "Pathway of the file"))
        self.pushButton_7.setText(_translate("MainWindow", "Run"))
        self.label_23.setText(_translate("MainWindow", "Output pathway"))
        self.label_24.setText(_translate("MainWindow", "Output pathway"))
        self.label_13.setText(_translate("MainWindow", "mgf file"))
        self.label_25.setText(_translate("MainWindow", "Output pathway"))
        self.label_26.setText(_translate("MainWindow", "Output pathway"))
        self.label_27.setText(_translate("MainWindow", "Label"))
        self.label_28.setText(_translate("MainWindow", "Label"))
        self.label_29.setText(_translate("MainWindow", "Label"))
        self.label_30.setText(_translate("MainWindow", "Training set ratio"))
        self.groupBox.setTitle(_translate("MainWindow", "Build MS/MS figures"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Data augmentation"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Data augmentation (Relative)"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Data augmentation (Absolute)"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Build labels"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Training Set"))
        self.label_5.setText(_translate("MainWindow", "Training Data"))
        self.label_6.setText(_translate("MainWindow", "Evaluation Data"))
        self.label_7.setText(_translate("MainWindow", "Batch Size"))
        self.label_16.setText(_translate("MainWindow", "Model Pathway"))
        self.label_17.setText(_translate("MainWindow", "Processing"))
        self.label_18.setText(_translate("MainWindow", "Information"))
        self.label_12.setText(_translate("MainWindow", "CNN model"))
        self.label_31.setText(_translate("MainWindow", "Classification"))
        self.label_32.setText(_translate("MainWindow", "Epoch"))
        self.pushButton_3.setText(_translate("MainWindow", "Run"))
        self.label_37.setText(_translate("MainWindow", "Learning rate"))
        self.groupBox_9.setTitle(_translate("MainWindow", "Parameters"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Model training"))
        self.pushButton_23.setText(_translate("MainWindow", "Test"))
        self.pushButton_24.setText(_translate("MainWindow", "Run"))
        self.label_21.setText(_translate("MainWindow", "Characteristic ions file"))
        self.label_22.setText(_translate("MainWindow", "Model pathway"))
        self.label_44.setText(_translate("MainWindow", "Classification"))
        self.label_45.setText(_translate("MainWindow", "Input mgf"))
        self.label_46.setText(_translate("MainWindow", "Input csv"))
        self.label_47.setText(_translate("MainWindow", "Input MS/MS"))
        self.label_73.setText(_translate("MainWindow", "Output pathway"))
        self.label_34.setText(_translate("MainWindow", "Output pathway"))
        self.groupBox_6.setTitle(_translate("MainWindow", "Checking characteristic ions"))
        self.pushButton_8.setText(_translate("MainWindow", "Run"))
        self.label_33.setText(_translate("MainWindow", "Input pathway"))
        self.groupBox_7.setTitle(_translate("MainWindow", "Building test data"))
        self.label_19.setText(_translate("MainWindow", "Input pathway"))
        self.label_20.setText(_translate("MainWindow", "Output pathway"))
        self.groupBox_8.setTitle(_translate("MainWindow", "Data testing"))
        self.label_35.setText(_translate("MainWindow", "CNN model"))
        self.label_74.setText(_translate("MainWindow", "Selected labels"))
        self.label_36.setText(_translate("MainWindow", "Filter threshold"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Data Testing"))
        self.actioninput.setText(_translate("MainWindow", "input"))


    def build_mgf(self):
        self.process_1 = Process1()
        self.process_1.signal1 = self.lineEdit.text()
        self.process_1.signal2 = self.lineEdit_5.text()
        self.process_1.daemon = True
        self.process_1.start()  # 开始进程

    def Data_augmentation(self):
        self.process_2 = Process2()
        self.process_2.signal3 = self.lineEdit_2.text()
        self.process_2.signal4 = self.lineEdit_4.text()
        self.process_2.signal5 = self.lineEdit_23.text()
        self.process_2.signal6 = self.lineEdit_17.text()
        self.process_2.daemon = True
        self.process_2.start()  # 开始进程

    def Data_augmentation_relative(self):
        self.process_3 = Process3()
        self.process_3.signal7 = self.lineEdit_9.text()
        self.process_3.signal8 = self.lineEdit_10.text()
        self.process_3.signal9 = self.lineEdit_21.text()
        self.process_3.signal10 = self.lineEdit_11.text()
        self.process_3.signal11 = self.lineEdit_18.text()
        self.process_3.daemon = True
        self.process_3.start()  # 开始进程

    def Data_augmentation_absolute(self):
        self.process_4 = Process4()
        self.process_4.signal12 = self.lineEdit_12.text()
        self.process_4.signal13 = self.lineEdit_13.text()
        self.process_4.signal14 = self.lineEdit_22.text()
        self.process_4.signal15 = self.lineEdit_19.text()
        self.process_4.daemon = True
        self.process_4.start()  # 开始进程

    def Build_label(self):
        self.thread_1 = Thread1()
        self.thread_1.signal_a = self.lineEdit_15.text()
        self.thread_1.signal_b = self.lineEdit_20.text()
        self.thread_1.start()  # 开始线程

    def call_back(self, value):
        self.progressBar.setProperty("value", float(value) * 100)

    def slot_text_browser(self, text):
        self.textBrowser.append(text)

    def Training_model(self):
        self.pushButton_3.setEnabled(False)
        self.thread_2 = Thread2()
        self.thread_2.signal_c = self.comboBox.currentText()
        self.thread_2.signal_d = self.lineEdit_6.text()
        self.thread_2.signal_e = self.lineEdit_25.text()
        self.thread_2.signal_f = self.lineEdit_26.text()
        self.thread_2.signal_g = self.lineEdit_27.text()
        self.thread_2.signal_h = self.lineEdit_28.text()
        self.thread_2.signal_i = self.lineEdit_29.text()
        self.thread_2.signal_z = self.lineEdit_30.text()
        self.thread_2.signal_A.connect(self.set_button)
        self.thread_2.signal_B.connect(self.call_back)
        self.thread_2.signal_C.connect(self.slot_text_browser)
        self.thread_2.start()  # 开始线程

    def set_button(self):
        self.pushButton_3.setEnabled(True)

    def searching_characteristic_ion(self):
        self.thread_3 = Thread3()
        self.thread_3.signal_j = self.lineEdit_7.text()
        self.thread_3.signal_k = self.lineEdit_14.text()
        self.thread_3.start()  # 开始线程

    def Building_evaluation_data(self):
        self.process_6 = Process6()
        self.process_6.signal_l = self.lineEdit_8.text()
        self.process_6.signal_m = self.lineEdit_16.text()
        self.process_6.daemon = True
        self.process_6.start()  # 开始线程

    def Testing(self):
        self.thread_5 = Thread5()
        self.thread_5.signal_n = self.comboBox_2.currentText()  # 模型选择
        self.thread_5.signal_o = self.lineEdit_40.text()  # 模型的位置
        self.thread_5.signal_p = self.lineEdit_45.text()  # 分类Classification
        self.thread_5.signal_q = self.lineEdit_46.text()  # mgf文件的位置 Input_mgf
        self.thread_5.signal_r = self.lineEdit_47.text()  # csv文件的位置 input_csv
        self.thread_5.signal_s = self.lineEdit_73.text()  # 裁剪好的图片和txt文件的总文件夹位置
        self.thread_5.signal_t = self.lineEdit_74.text()  # 输出的mgf和csv文件的位置
        self.thread_5.signal_u = self.lineEdit_75.text()  # 特征碎片离子的csv文件
        self.thread_5.signal_v = self.lineEdit_76.text()  # 输入要筛选的化合物对应的标签
        self.thread_5.signal_w = self.lineEdit_77.text()  # 碎片离子匹配的阈值
        # self.thread_5.signal_u = self.lineEdit_75.text()  # 上面识别整理好的特征例子的文件位置
        self.thread_5.start()  # 开始线程


class Thread1(QThread):
    signal_a = pyqtSignal(str)
    signal_b = pyqtSignal(str)

    def __init__(self):
        super(Thread1, self).__init__()

    def run(self):
        Dataset_Pathway = self.signal_a
        Training_ratio = self.signal_b
        Dataset_labels(Dataset_Pathway, Training_ratio)


'''模型训练，从这里开始'''


class Thread2(QThread):
    signal_c = pyqtSignal(str)  # 小写字母传框里的信息
    signal_d = pyqtSignal(str)
    signal_e = pyqtSignal(int)
    signal_f = pyqtSignal(str)
    signal_g = pyqtSignal(str)
    signal_h = pyqtSignal(int)
    signal_i = pyqtSignal(str)
    signal_z = pyqtSignal(str)
    signal_A = pyqtSignal()  # 大写字母传特殊的信息
    signal_B = pyqtSignal(str)
    signal_C = pyqtSignal(str)

    def __init__(self):
        super(Thread2, self).__init__()

    def run(self):
        b = self.signal_h  # 疑点
        # a = batch_size  # 疑点
        batch_size = 1 * int(b)  # 如果直接把batch_size传进去DataLoader会报错

        # print(batch_size)

        # # 给训练集和测试集分别创建一个数据集加载器
        training_set = self.signal_f
        train_data = LoadData(
            training_set + "/train.txt", True)
        test_set = self.signal_g
        valid_data = LoadData(
            test_set + "/test.txt", False)

        train_dataloader = DataLoader(dataset=train_data, num_workers=8, pin_memory=True, batch_size=batch_size,
                                      shuffle=True)
        test_dataloader = DataLoader(dataset=valid_data, num_workers=8, pin_memory=True, batch_size=batch_size)

        for X, y in test_dataloader:
            print("Shape of X [N, C, H, W]: ", X.shape)
            self.signal_C.emit("Shape of X [N, C, H, W]: " + ' ' + str(X.shape))
            print("Shape of y: ", y.shape, y.dtype)
            self.signal_C.emit("Shape of y: " + ' ' + str(y.shape) + ' ' + str(y.dtype))
            break

        # 如果显卡可用，则用显卡进行训练
        device = "cuda" if cuda.is_available() else "cpu"
        print("Using {} device".format(device))
        self.signal_C.emit(str("Using {} device".format(device)))

        # 调用刚定义的模型，将模型转到GPU（如果可用）
        classification_number = self.signal_d
        classification = classification_number
        if self.signal_c == 'NeuralNetwork1':
            model = Training_model.NeuralNetwork1(int(classification)).to(device)
            print(model)
            self.signal_C.emit(str(model))
        if self.signal_c == 'NeuralNetwork2':
            model = Training_model.NeuralNetwork2(int(classification)).to(device)
            print(model)
            self.signal_C.emit(str(model))
        if self.signal_c == 'NeuralNetwork3':
            model = Training_model.NeuralNetwork3(int(classification)).to(device)
            print(model)
            self.signal_C.emit(str(model))



        # 定义损失函数，计算相差多少，交叉熵，
        loss_fn = nn.CrossEntropyLoss()

        # 定义优化器，用来训练时候优化模型参数，随机梯度下降法
        # lR = 1 * 1e-4
        epoch = self.signal_e
        epochs = epoch
        LR = self.signal_z
        print(float(LR))
        optimizer = SGD(model.parameters(), lr=float(LR), momentum=0.5, weight_decay=1 * 1e-5)  # 初始学习率
        for x in range(int(epoch)):
            if (x+1) % 5 == 0:
                for parameter in optimizer.param_groups:
                    parameter['lr'] *= 0.9

        # 一共训练5次

        best = 0.0
        Model_pathway = self.signal_i
        for t in range(int(epochs)):
            process = str(float((t+1) / int(epochs)))
            print(f"Epoch {t + 1}\n-------------------------------")
            self.signal_C.emit(str(f"Epoch {t + 1}\n-------------------------------"))
            train_loss = Training_model.train(train_dataloader, model, loss_fn, optimizer, batch_size)
            self.signal_C.emit(str("average_loss: " + ' ' + train_loss))
            accuracy, avg_loss, b, c = Training_model.test(test_dataloader, model, loss_fn)
            self.signal_C.emit(str(b))
            self.signal_C.emit(str(c))
            Training_model.write_result(
                Model_pathway + "/traindata.txt",
                t + 1, train_loss, avg_loss, accuracy)
            save(model.state_dict(),
                       Model_pathway + "/resnet_epoch_" + str(
                           t + 1) + "_acc_" + str(accuracy) + ".pth")
            self.signal_B.emit(process)
        self.signal_A.emit()  # 发射信息


'''模型训练，到这里结束'''


'''以下是寻找每类化合物的特征离子'''


class Thread3(QThread):
    signal_j = pyqtSignal(str)
    signal_k = pyqtSignal(str)

    def __init__(self):
            super(Thread3, self).__init__()

    def run(self):
        input_mgf = self.signal_j
        output_path = self.signal_k
        CI_search(input_mgf, output_path)


'''以下是构建测试数据的图片和标签'''


class Process6(Process):
    signal_l = pyqtSignal(str)
    signal_m = pyqtSignal(str)

    def __init__(self):
        super(Process6, self).__init__()

    def run(self):
        input_pathway = self.signal_l
        output_pathway = self.signal_m
        # result_path = output_pathway + '/' + 'result'
        # mkdri(result_path)
        cropped_output_path = output_pathway + '/' + 'test_cropped'
        mkdri(cropped_output_path)
        eval_path = output_pathway + '/' + 'eval'
        mkdri(eval_path)
        Build_evaluate_figure(input_pathway, cropped_output_path)
        CreateEvalData(cropped_output_path, eval_path)


'''以下是测试的部分'''


class Thread5(QThread):
    signal_n = pyqtSignal(str)
    signal_o = pyqtSignal(str)
    signal_p = pyqtSignal(str)
    signal_q = pyqtSignal(str)
    signal_r = pyqtSignal(str)
    signal_s = pyqtSignal(str)
    signal_t = pyqtSignal(str)
    signal_u = pyqtSignal(str)
    signal_v = pyqtSignal(str)
    signal_w = pyqtSignal(str)

    def __init__(self):
            super(Thread5, self).__init__()

    def run(self):
        classification_number = self.signal_p
        model_loc = self.signal_o  ###填入神经网络的参数文件
        input_MSMS = self.signal_s  # 填入转换好的图片和txt
        input_mgf = self.signal_q
        input_csv = self.signal_r
        output_path = self.signal_t
        characteristic_ion_file = self.signal_u
        Labels_list = self.signal_v
        filter_threshold = self.signal_w

        """ 1. 导入模型结构"""

        # print(str(characteristic_ion_file))
        global model_test, row_ID, row_mass, row_retention_time, Peak_area, Peak_Unnamed, titles
        device = "cuda" if cuda.is_available() else "cpu"
        # classification_number = self.signal_p
        if self.signal_n == 'NeuralNetwork1':
            model_test = Training_model.NeuralNetwork1(int(classification_number)).to(device)
        if self.signal_n == 'NeuralNetwork2':
            model_test = Training_model.NeuralNetwork2(int(classification_number)).to(device)
        if self.signal_n == 'NeuralNetwork3':
            model_test = Training_model.NeuralNetwork3(int(classification_number)).to(device)

        ''' 2. 加载模型参数 '''

        # model_loc = self.signal_o  ###填入神经网络的参数文件
        model_dict = load(model_loc)
        model_test.load_state_dict(model_dict)
        model = model_test.to(device)

        ''' 3. 加载图片 '''

        # input_MSMS = self.signal_s  # 填入转换好的图片和txt
        valid_data = LoadData(input_MSMS + "/eval/eval.txt", train_flag=False)
        # 填入验证集的TXT文件
        test_dataloader = DataLoader(dataset=valid_data, num_workers=4, pin_memory=False, batch_size=1)

        ''' 4. 获取结果'''
        label_names = []  # 化合物分类，注意排序
        label_list, likelihood_list = eval(test_dataloader, model)
        for class_num in range(int(classification_number)):
            label_names.append(class_num)

        result_names = [label_names[i] for i in label_list]
        list = [result_names, likelihood_list]
        df = pd.DataFrame(data=list)
        # print(df)
        df2 = pd.DataFrame(df.values.T, columns=["label", "likelihood"])
        # print(df2)
        df2.to_csv(input_MSMS + '/' + 'testdata.csv', encoding='gbk')
        # 这里可以输出预测的结果，这里的预测结果的顺序独赢的eval.txt上的排列顺序，txt上的排列顺序与csv上的排列顺序不同，因此需要进一步的处理
        # print(label_list)

        path = input_MSMS + '/' + 'test_cropped'  ##转化的质谱图文件的位置
        name_index = []
        labels = []  # 记录标签的信息

        '''下面测试try'''
        try:
            labels_list = Labels_list.split(',')
            selected_labels_list = []
            for labels_2 in labels_list:
                selected_labels_list.append(int(labels_2))
            print(selected_labels_list)
            for i, label, in enumerate(label_list):
                # print(label)
                # if label == 3:
                '''标签的参数也需要定制'''
                if label in selected_labels_list:  # 这里一定要用==号，不能够用=号，=号是赋值的意思，这里的lable是化合物种类对应的编号，0对应other compound
                    if likelihood_list[i] > 0.9:
                        labels.append(label)  # 这个标签就是最终输出的化合物的标签，不需要考虑顺序问题
                        name_index.append(i)  ##乌药烷在第几个

            # print('name_index:', name_index)
            # mgf_file = input_mgf

            in_source = []
            unname1 = []
            # input_mgf = self.signal_q
            # with mgf.read(input_mgf) as spectra:  # 用MZmine2生成的mgf文件
            for a, b, c in os.walk(path):
                name_list = []
                name_list_num = []
                for names in c:
                    name = os.path.splitext(names)  # 也是按首数字排列
                    name_list.append(name[0])  # title的总列表
                    name_list_num.append(int(name[0]))
                # print('name_list:', name_list)
                name_list_num.sort()  # 对列表中的数字进行排序，从小到大（升序）
                # print('name_list_num', name_list_num)
                titles = []
                titles_num = []
                for index in name_index:
                    titles.append(name_list[index])  ###挑出的title的列表
                    titles_num.append(int(name_list[index]))


            # input_csv = self.signal_r
            with open(input_csv) as f:
                reader = csv.reader(f)
                header_row = next(reader)[3]  # 读取第三列的表头
            data = pd.read_csv(input_csv)
            # background_y = data['row m/z']
            # background_x = data['row retention time']

            row_ID = []
            row_mass = []
            row_retention_time = []
            Peak_area = []
            Peak_Unnamed = []
            characteristic_ion_list = []

            title_index = map(name_list_num.index, titles_num)  ## 这句代码的意思是找title_num中每一个元素在在name_list中的下标
            for num in title_index:
                row_ID.append(int(data.loc[num]['row ID']))
                row_mass.append(data.loc[num]['row m/z'])
                row_retention_time.append(data.loc[num]['row retention time'])
                Peak_area.append(data.loc[num][str(header_row)])
                Peak_Unnamed.append(data.loc[num]['Unnamed: 4'])
            # print('row_ID:', row_ID)

            for a, m in enumerate(row_retention_time):
                for n in range(a + 1, len(row_retention_time)):
                    if m == row_retention_time[n]:
                        if row_mass[a] > row_mass[n]:
                            if not int(row_mass[a]) - int(
                                    row_mass[n]) == 5:  ##排除源内裂解，把同一保留时间的化合物质量相比较，去掉最小的化合物，但是要保留M+NH4的，反而要排除M+Na的。
                                # if int(row_mass[a]) - int(row_mass[n]) == 162 or int(row_mass[a]) - int(row_mass[n]) == 18 \
                                #         or int(row_mass[a]) - int(row_mass[n]) == 17:
                                if n not in in_source:
                                    in_source.append(n)
                            if int(row_mass[a]) - int(row_mass[n]) == 5:
                                if a not in in_source:
                                    in_source.append(a)
                        elif row_mass[a] < row_mass[n]:
                            if not int(row_mass[n]) - int(row_mass[a]) == 5:
                                # if int(row_mass[n]) - int(row_mass[a]) == 162 or int(row_mass[n]) - int(row_mass[a]) == 18 \
                                #         or int(row_mass[n]) - int(row_mass[a]) == 17:
                                if a not in in_source:
                                    in_source.append(a)
                            if int(row_mass[n]) - int(row_mass[a]) == 5:
                                if a not in in_source:
                                    in_source.append(n)

            for title in row_ID:
                spectrum = mgf.get_spectrum(input_mgf, title=str(int(title)))
                # params = spectrum.get('params')
                # mass = params['pepmass'][0]
                # print(mass)
                if max(spectrum['intensity array']) < 1000:  # 排除噪声（最大的碎片强度小于1000）
                    unname1.append(title)
                # print('unname1', unname1)
                unname1_index = map(row_ID.index, unname1)
                for index1 in unname1_index:
                    if index1 not in in_source:
                        in_source.append(index1)
            in_source.sort()
            for z in in_source[::-1]:  # 到这遍历列表的一种方式
                row_ID.pop(z)
                row_mass.pop(z)
                row_retention_time.pop(z)
                Peak_area.pop(z)
                Peak_Unnamed.pop(z)
                titles.pop(z)
                labels.pop(z)
                # print(len(row_ID))
                # print(len(labels))
        except:
            pass
        '''try 测试到此结束'''

        if os.path.exists(str(characteristic_ion_file)):
            print(labels)
            print('yes')
            data = pd.read_csv(characteristic_ion_file)
            for ci in range(1, int(classification_number)):
                a = str(ci)
                characteristic_ion_list.append(data[a])

            # print(characteristic_ion_list)
            filtered_list = []
            # for h, compound_labels in enumerate(labels):  # 有问题
            # with mgf.read(input_mgf) as spectra: #  有问题
            #     for i, spectrum in enumerate(spectra):

            for h, title in enumerate(titles):
                spectrum = mgf.get_spectrum(input_mgf, title=str(int(title)))

                selected_ions = []
                selected_ions_index = []
                intensity_max_50 = heapq.nlargest(20, spectrum['intensity array'])
                newlist = spectrum['intensity array'].tolist()
                newlist2 = spectrum['m/z array'].tolist()
                # print(newlist)
                # max_50_index = map(newlist.index, intensity_max_50)

                for num in intensity_max_50:
                    f = newlist.index(num)
                    selected_ions.append(int(spectrum['m/z array'][f]))
                    newlist.pop(f)
                    newlist2.pop(f)

                    # selected_ions_index.append(f)

                # print(max_50_index)
                # for j in max_50_index:
                #     print(j)
                #     selected_ions.append(int(spectrum['m/z array'][j]))

                # print(selected_ions)
                score = 0
                for k in selected_ions:
                    # print(compound_labels)
                    if k in characteristic_ion_list[(labels[h] - 1)]:
                        score = score + 1
                u = characteristic_ion_list[(labels[h] - 1)]
                # print(score)
                # print(u)
                print(score)
                # print('length:', len(u))
                if score / 20 > float(filter_threshold):   # 这个参数需要定制
                    filtered_list.append(h)

                print(filtered_list)

                intensity_max = max(spectrum['intensity array'])  # intensity normalization
                intensity_min = min(spectrum['intensity array'])  # intensity normalization
                x = (spectrum['intensity array'] - intensity_min) / (intensity_max - intensity_min)
                intensity_score = 0
                for intensity in x:
                    if intensity > 0.2:
                        intensity_score = intensity_score + 1
                if intensity_score < 5:
                    if h not in filtered_list:
                        filtered_list.append(h)  # 这部分代码是为了排除M+Na

            for m in filtered_list[::-1]:  # 到这遍历列表的一种方式
                row_ID.pop(m)
                row_mass.pop(m)
                row_retention_time.pop(m)
                Peak_area.pop(m)
                Peak_Unnamed.pop(m)
                titles.pop(m)

            list_total = [row_ID, row_mass, row_retention_time, Peak_area, Peak_Unnamed]
            df = pd.DataFrame(data=list_total)
            df2 = pd.DataFrame(df.values.T,
                               columns=["row ID", "row m/z", 'row retention time', 'Peak area', 'Unnamed: 4'])
            df2.to_csv(output_path + '/' + 'target.csv', encoding='gbk', index=False)

            with mgf.read(input_mgf) as spectra:
                for title in titles:
                    spectrum = mgf.get_spectrum(input_mgf, title=str(int(title)))
                    mgf.write((spectrum,),
                              output=output_path + '/' + 'target.mgf')  # 识别后输出的mgf

        else:  # 下面是原来的
            try:
                '''这个部分是不筛选化合物然后直接写入csv和mgf文件的'''
                list_total = [row_ID, row_mass, row_retention_time, Peak_area, Peak_Unnamed]
                df = pd.DataFrame(data=list_total)
                df2 = pd.DataFrame(df.values.T, columns=["row ID", "row m/z", 'row retention time', 'Peak area', 'Unnamed: 4'])
                df2.to_csv(output_path + '/' + 'shuchu.csv', encoding='gbk', index=False)
                ##这里填写识别后生成的csv文件

                with mgf.read(input_mgf) as spectra:
                    for title in titles:
                        spectrum = mgf.get_spectrum(input_mgf, title=str(int(title)))
                        mgf.write((spectrum,),
                                  output=output_path + '/' + 'target.mgf')  # 识别后输出的mgf
            except:
                pass


class Process1(Process):  # 进程1
    signal1 = pyqtSignal(str)
    signal2 = pyqtSignal(str)

    def __init__(self):
        super(Process1, self).__init__()

    def run(self):
        openfile_name = self.signal1
        output_path = self.signal2
        cropped_output_path = self.signal2
        Build_MS_Figure(openfile_name, output_path, cropped_output_path)


class Process2(Process):  # 线程2
    signal3 = pyqtSignal(str)
    signal4 = pyqtSignal(str)
    signal5 = pyqtSignal(str)
    signal6 = pyqtSignal(str)

    def __init__(self):
        super(Process2, self).__init__()

    def run(self):
        mgf_file = self.signal3
        rounds = self.signal4  # self.lineEdit_4.text()输出的是字符串str类型
        label = self.signal5
        output_pathway = self.signal6
        Data_augmentation(mgf_file, rounds, output_pathway, label)


class Process3(Process):  # 线程3
    signal7 = pyqtSignal(str)
    signal8 = pyqtSignal(str)
    signal9 = pyqtSignal(str)
    signal10 = pyqtSignal(str)
    signal11 = pyqtSignal(str)

    def __init__(self):
        super(Process3, self).__init__()

    def run(self):
        mgf_file = self.signal7
        rounds = self.signal8  # self.lineEdit_4.text()输出的是字符串str类型
        label = self.signal9
        level = self.signal10
        output_pathway = self.signal11
        Data_augmentation_relative(mgf_file, level, rounds, output_pathway, label)


class Process4(Process):  # 进程4
    signal12 = pyqtSignal(str)
    signal13 = pyqtSignal(str)
    signal14 = pyqtSignal(str)
    signal15 = pyqtSignal(str)

    def __init__(self):
        super(Process4, self).__init__()

    def run(self):
        mgf_file = self.signal12
        rounds = self.signal13  # self.lineEdit_4.text()输出的是字符串str类型
        label = self.signal14
        output_pathway = self.signal15
        Data_augmentation_absolute(mgf_file, rounds, output_pathway, label)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    p1 = Process1()
    ui.pushButton.clicked.connect(lambda: p1)
    p2 = Process2()
    ui.pushButton_2.clicked.connect(lambda: p2)
    # p2 = Process2()
    # ui.pushButton_2.clicked.connect(lambda: p2)
    p3 = Process3()
    ui.pushButton_5.clicked.connect(lambda: p3)
    p4 = Process4()
    ui.pushButton_6.clicked.connect(lambda: p4)
    MainWindow.show()

    sys.exit(app.exec_())
    # os.system('pause')  # 这句话才是重点！


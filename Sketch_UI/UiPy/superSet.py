from PyQt6 import QtCore, QtGui, QtWidgets
import sys
import json
import os

def load_data(brick_database=["regular_plate.json"]):
    data = []
    for data_base in brick_database:
        database_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "bricks_modeling",
                                     "database", data_base)
        with open(database_file) as f:
            temp = json.load(f)
            data.extend(temp)
    return data


class Super_Set(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(537, 366)
        self.toolButton = QtWidgets.QToolButton(Dialog)
        self.toolButton.setGeometry(QtCore.QRect(330, 40, 41, 31))
        self.toolButton.setObjectName("toolButton")
        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setGeometry(QtCore.QRect(30, 40, 291, 31))
        self.comboBox.setObjectName("comboBox")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(30, 20, 191, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(30, 80, 171, 16))
        self.label_2.setObjectName("label_2")
        self.listWidget = QtWidgets.QListWidget(Dialog)
        self.listWidget.setGeometry(QtCore.QRect(30, 100, 281, 241))
        self.listWidget.setObjectName("listWidget")
        data = load_data()
        for i in range(len(data)):
            item = QtWidgets.QListWidgetItem(data[i]["id"])
            self.listWidget.addItem(item)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(390, 300, 121, 41))
        self.pushButton.setObjectName("pushButton")
        self.listWidget_2 = QtWidgets.QListWidget(Dialog)
        self.listWidget_2.setGeometry(QtCore.QRect(390, 100, 121, 192))
        self.listWidget_2.setObjectName("listWidget_2")
        self.pushButton_preview = QtWidgets.QPushButton(Dialog)
        self.pushButton_preview.setGeometry(QtCore.QRect(320, 110, 61, 31))
        self.pushButton_preview.setObjectName("pushButton_preview")
        self.pushButton_add = QtWidgets.QPushButton(Dialog)
        self.pushButton_add.setGeometry(QtCore.QRect(320, 150, 61, 31))
        self.pushButton_add.setObjectName("pushButton_add")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(400, 80, 171, 16))
        self.label_3.setObjectName("label_3")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.toolButton.setText(_translate("Dialog", "..."))
        self.label.setText(_translate("Dialog", "Choose the basement file..."))
        self.label_2.setText(_translate("Dialog", "Choose the brick set..."))
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        self.listWidget.setSortingEnabled(__sortingEnabled)
        self.pushButton.setText(_translate("Dialog", "Generate"))
        __sortingEnabled = self.listWidget_2.isSortingEnabled()
        self.listWidget_2.setSortingEnabled(False)
        self.listWidget_2.setSortingEnabled(__sortingEnabled)
        self.pushButton_preview.setText(_translate("Dialog", "Preview"))
        self.pushButton_add.setText(_translate("Dialog", "Add"))
        self.label_3.setText(_translate("Dialog", "Current Brick Set"))

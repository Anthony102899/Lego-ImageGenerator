from PyQt6 import QtCore, QtGui, QtWidgets


class Dismap(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(387, 375)
        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setGeometry(QtCore.QRect(50, 30, 221, 31))
        self.comboBox.setObjectName("comboBox")
        self.toolButton = QtWidgets.QToolButton(Dialog)
        self.toolButton.setGeometry(QtCore.QRect(280, 30, 51, 31))
        self.toolButton.setObjectName("toolButton")
        self.imglabel = QtWidgets.QLabel(Dialog)
        self.imglabel.setGeometry(QtCore.QRect(50, 80, 281, 231))
        self.imglabel.setText("")
        self.imglabel.setObjectName("imglabel")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(70, 320, 231, 41))
        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Choose the image to process..."))
        self.toolButton.setText(_translate("Dialog", "..."))
        self.pushButton.setText(_translate("Dialog", "Generate Distance Map"))

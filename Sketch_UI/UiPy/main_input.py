from PyQt6 import QtCore, QtGui, QtWidgets


class MainInputDialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(396, 203)
        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setGeometry(QtCore.QRect(140, 20, 191, 31))
        self.comboBox.setObjectName("comboBox")
        self.toolButton = QtWidgets.QToolButton(Dialog)
        self.toolButton.setGeometry(QtCore.QRect(340, 20, 41, 31))
        self.toolButton.setObjectName("toolButton")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 20, 71, 31))
        self.label.setStyleSheet("font-size:14px\n"
"")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(20, 90, 111, 16))
        self.label_2.setStyleSheet("font-size: 14px")
        self.label_2.setObjectName("label_2")
        self.spinBox = QtWidgets.QSpinBox(Dialog)
        self.spinBox.setGeometry(QtCore.QRect(140, 80, 41, 31))
        self.spinBox.setObjectName("spinBox")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(290, 160, 75, 24))
        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Choose input file"))
        self.toolButton.setText(_translate("Dialog", "..."))
        self.label.setText(_translate("Dialog", "Input (.pkl)"))
        self.label_2.setText(_translate("Dialog", "Layer Number"))
        self.pushButton.setText(_translate("Dialog", "Next"))

from PyQt6 import QtCore, QtGui, QtWidgets


class precPage(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(581, 346)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(50, 50, 221, 111))
        self.pushButton.setStyleSheet("font-size: 16px\n"
"")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(310, 50, 221, 111))
        self.pushButton_2.setStyleSheet("font-size: 16px\n"
"")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(50, 190, 221, 111))
        self.pushButton_3.setStyleSheet("font-size: 16px")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Dialog)
        self.pushButton_4.setGeometry(QtCore.QRect(310, 190, 221, 111))
        self.pushButton_4.setStyleSheet("font-size: 16px")
        self.pushButton_4.setObjectName("pushButton_4")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "Generate Distance Map"))
        self.pushButton_2.setText(_translate("Dialog", "Precompute For Layers"))
        self.pushButton_3.setText(_translate("Dialog", "Generate Super Set"))
        self.pushButton_4.setText(_translate("Dialog", "Generate Adjacency Graph"))

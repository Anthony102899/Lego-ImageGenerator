from PyQt6 import QtCore, QtGui, QtWidgets


class precPage(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(50, 30, 291, 91))
        self.pushButton.setStyleSheet("font-size: 16px\n"
"")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(50, 160, 291, 101))
        self.pushButton_2.setStyleSheet("font-size: 16px\n"
"")
        self.pushButton_2.setObjectName("pushButton_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Choose Precompute Operation"))
        self.pushButton.setText(_translate("Dialog", "Produce Distance Map"))
        self.pushButton_2.setText(_translate("Dialog", "Produce Precompute Pickle File"))

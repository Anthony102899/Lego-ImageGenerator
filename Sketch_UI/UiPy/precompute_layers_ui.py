from PyQt6 import QtCore, QtGui, QtWidgets

class PrecumputeLayersUi(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(387, 405)
        self.graph_select_box = QtWidgets.QComboBox(Dialog)
        self.graph_select_box.setGeometry(QtCore.QRect(50, 30, 221, 31))
        self.graph_select_box.setObjectName("graph_select_box")

        self.open_graph_button = QtWidgets.QToolButton(Dialog)
        self.open_graph_button.setGeometry(QtCore.QRect(280, 30, 51, 31))
        self.open_graph_button.setObjectName("open_graph_button")

        self.image_select_box = QtWidgets.QComboBox(Dialog)
        self.image_select_box.setGeometry(QtCore.QRect(50, 80, 221, 31))
        self.image_select_box.setObjectName("image_select_box")

        self.open_image_button = QtWidgets.QToolButton(Dialog)
        self.open_image_button.setGeometry(QtCore.QRect(280, 80, 51, 31))
        self.open_image_button.setObjectName("open_image_button")

        self.image_label = QtWidgets.QLabel(Dialog)
        self.image_label.setGeometry(QtCore.QRect(50, 120, 281, 231))
        self.image_label.setText("")
        self.image_label.setObjectName("image_label")

        self.push_button = QtWidgets.QPushButton(Dialog)
        self.push_button.setGeometry(QtCore.QRect(70, 360, 231, 41))
        self.push_button.setObjectName("push_button")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Choose the graph and image to process"))
        self.open_graph_button.setText(_translate("Dialog", "Graph"))
        self.open_image_button.setText(_translate("Dialog", "Image"))
        self.push_button.setText(_translate("Dialog", "Generate Precomputed Model"))




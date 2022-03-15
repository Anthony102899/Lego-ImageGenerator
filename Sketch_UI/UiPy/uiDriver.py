from pklFilePage import *
from welcomePage import *
from imageUpload import *
from layer import *
from precPage import *
from PyQt6.QtWidgets import QApplication, QMainWindow, QDialog
from Sketch_UI.UiPy.dismap import *
import sys

from solvers.generation_solver.distance_map import *
from solvers.generation_solver.precompute import *


class parentWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.main_ui = Ui_MainWindow()
        self.main_ui.setupUi(self)


class childWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.child = Pkl_Dialog()
        self.child.setupUi(self)

    def open_file(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                   "All Files(*);;Pickle Files(*.pkl)")
        self.fileName = fileName
        self.fileType = fileType
        self.updateComboBox()
        print(fileName)
        print(fileType)

    def updateComboBox(self):
        self.child.comboBox.addItem(self.fileName)


class precWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.ui = precPage()
        self.ui.setupUi(self)


class dismapWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.fileName = ""
        self.fileType = ""
        self.ui = Dismap()
        self.ui.setupUi(self)
        img = QtGui.QPixmap('../resource/icon/upload.jpg')
        self.ui.imglabel.setPixmap(img)
        self.ui.imglabel.setScaledContents(True)

    def open_file(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                   "All Files(*);;jpeg Files(*.jpg);;png Files(*.png)")
        self.fileName = fileName
        self.fileType = fileType
        img = QtGui.QPixmap(self.fileName)
        self.ui.imglabel.setPixmap(img)
        self.ui.imglabel.setScaledContents(True)
        self.updateComboBox()

    def generate_dismap(self):
        file_path = self.fileName
        distance_map = DistanceMap(file_path, 2)
        distance_map.generate_distance_map(2)
        self.close()

    def updateComboBox(self):
        self.ui.comboBox.addItem(self.fileName)


class imgUploadWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.child = upload_Dialog()
        self.child.setupUi(self)
        img = QtGui.QPixmap('../resource/icon/upload.jpg')
        self.child.label_2.setPixmap(img)
        self.child.label_2.setScaledContents(True)

    def open_file(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                   "All Files(*);;jpeg Files(*.jpg);;png Files(*.png)")
        self.fileName = fileName
        self.fileType = fileType
        img = QtGui.QPixmap(self.fileName)
        self.child.label_2.setPixmap(img)
        self.child.label_2.setScaledContents(True)
        print(fileName)
        print(fileType)


class LayerWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.child = Layer_Dialog()
        self.child.setupUi(self)
        img1 = QtGui.QPixmap('../resource/images/LEGO6_1.png')
        self.child.label.setPixmap(img1)
        self.child.label.setScaledContents(True)
        img2 = QtGui.QPixmap('../resource/images/LEGO6_2.png')
        self.child.label_2.setPixmap(img2)
        self.child.label_2.setScaledContents(True)
        img3 = QtGui.QPixmap('../resource/images/LEGO6_3.png')
        self.child.label_3.setPixmap(img3)
        self.child.label_3.setScaledContents(True)
        img4 = QtGui.QPixmap('../resource/images/LEGO6_4.png')
        self.child.label_4.setPixmap(img4)
        self.child.label_4.setScaledContents(True)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = parentWindow()
    prec = precWindow()
    child = childWindow()
    imgUpload = imgUploadWindow()
    layer = LayerWindow()
    dismap = dismapWindow()
    # bind button event
    btn_start = window.main_ui.pushButton_2  # The start button on main window
    btn_start.clicked.connect(child.show)

    btn_prec = window.main_ui.pushButton  # The precompute button on main window
    btn_prec.clicked.connect(prec.show)

    btn_precompute = prec.ui.pushButton_2  # The precompute button on precompute window
    btn_precompute.clicked.connect(Precompute)

    btn_dismap = prec.ui.pushButton  # The distance map button on precompute window
    btn_dismap.clicked.connect(dismap.show)

    btn_dismap_choose = dismap.ui.toolButton  # The distance map generation button on dismap window
    btn_dismap_choose.clicked.connect(dismap.open_file)

    btn_dismap_gen = dismap.ui.pushButton
    btn_dismap_gen.clicked.connect(dismap.generate_dismap)

    btn_choose_pkl = child.child.toolButton  # The open file button on child window
    btn_choose_pkl.clicked.connect(child.open_file)

    btn_import_pkl = child.child.pushButton
    btn_import_pkl.clicked.connect(imgUpload.show)

    btn_choose_img = imgUpload.child.pushButton
    btn_choose_img.clicked.connect(imgUpload.open_file)

    btn_parse_img = imgUpload.child.pushButton_2
    btn_parse_img.clicked.connect(layer.show)

    window.show()
    sys.exit(app.exec())

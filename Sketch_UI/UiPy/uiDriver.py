from welcomePage import *
from PyQt6.QtWidgets import QApplication, QMainWindow, QDialog
from Sketch_UI.UiPy.dismap import *
from Sketch_UI.UiPy.precPage import *
from Sketch_UI.UiPy.superSet import *
import sys
import os
import json

from solvers.generation_solver.distance_map import *
from solvers.generation_solver.precompute import *


class parentWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.main_ui = Ui_MainWindow()
        self.main_ui.setupUi(self)


class precWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.ui = precPage()
        self.ui.setupUi(self)

    def precompute(self):
        try:
            Precompute()
        except Exception as e:
            print("error!")


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


class Super_set_window(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.ui = Super_Set()
        self.ui.setupUi(self)
        self.set_selected = []

    def add(self):
        try:
            item = self.ui.listWidget.currentItem()
            brick = item.text()
            if self.set_selected.__contains__(brick):
                return
            self.set_selected.append(brick)
            self.ui.listWidget_2.addItem(brick)
            print(self.set_selected)
        except Exception as e:
            print(e)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = parentWindow()
    prec = precWindow()
    dismap = dismapWindow()
    superset = Super_set_window()
    # bind button event
    btn_start = window.main_ui.pushButton_2  # The start button on main window
    # TODO: add btn_Start function

    btn_prec = window.main_ui.pushButton  # The precompute button on main window
    btn_prec.clicked.connect(prec.show)

    btn_precompute = prec.ui.pushButton_2  # The precompute button on precompute window
    btn_precompute.clicked.connect(prec.precompute)

    btn_dismap = prec.ui.pushButton  # The distance map button on precompute window
    btn_dismap.clicked.connect(dismap.show)

    btn_dismap_choose = dismap.ui.toolButton  # The distance map generation button on dismap window
    btn_dismap_choose.clicked.connect(dismap.open_file)

    btn_dismap_gen = dismap.ui.pushButton
    btn_dismap_gen.clicked.connect(dismap.generate_dismap)

    btn_superset = prec.ui.pushButton_3  # Show the Superset by pushing super set button in precompute page
    btn_superset.clicked.connect(superset.show)

    btn_add_brick = superset.ui.pushButton_add  # Add selected brick to selected brick set
    btn_add_brick.clicked.connect(superset.add)

    window.show()
    sys.exit(app.exec())

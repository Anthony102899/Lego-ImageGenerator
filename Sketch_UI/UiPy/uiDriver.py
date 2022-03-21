import sys
import os
import json
import open3d

from welcomePage import *
from PyQt6.QtWidgets import QApplication, QMainWindow, QDialog
from Sketch_UI.UiPy.dismap import *
from Sketch_UI.UiPy.precPage import *
from Sketch_UI.UiPy.superSet import *
from solvers.generation_solver.distance_map import *
from solvers.generation_solver.precompute import *
from solvers.generation_solver.gen_sketch_placement import *

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
        self.update_combobox()

    def generate_dismap(self):
        file_path = self.fileName
        distance_map = DistanceMap(file_path, 2)
        distance_map.generate_distance_map(2)
        self.hide()

    def update_combobox(self):
        self.ui.comboBox.addItem(self.fileName)


class Super_set_window(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.ui = Super_Set()
        self.ui.setupUi(self)
        self.set_selected = []
        self.fileName = ""
        self.fileType = ""

    def open_file(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                   "LDraw Files(*.ldr)")
        self.fileName = fileName
        self.fileType = fileType
        self.update_combobox()

    def update_combobox(self):
        self.ui.comboBox.addItem(self.fileName)

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

    def run(self):
        print(self.set_selected)
        try:
            generate_superset(self.set_selected, self.fileName)
        except Exception as e:
            print(e)
        self.hide()
0
    # def visualize(self):
    # TODO: use open3d.mesh to visualize the 3d model of bricks


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

    btn_superset_tb = superset.ui.toolButton  # Select basement file by click the tool button
    btn_superset_tb.clicked.connect(superset.open_file)

    btn_add_brick = superset.ui.pushButton_add  # Add selected brick to selected brick set
    btn_add_brick.clicked.connect(superset.add)

    btn_preview = superset.ui.pushButton_preview
    # btn_preview.clicked.connect(superset.visualize)

    btn_generate_superset = superset.ui.pushButton
    btn_generate_superset.clicked.connect(superset.run)

    window.show()
    sys.exit(app.exec())

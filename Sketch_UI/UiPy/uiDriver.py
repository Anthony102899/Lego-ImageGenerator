import sys
import os
import json
import open3d

from welcomePage import *
from PyQt6.QtWidgets import QApplication, QMainWindow, QDialog
from multiprocessing import Process
from Sketch_UI.UiPy.dismap import *
from Sketch_UI.UiPy.precPage import *
from Sketch_UI.UiPy.superSet import *
from Sketch_UI.UiPy.adjacency_graph_ui import *
from Sketch_UI.UiPy.main_solver import *
from Sketch_UI.UiPy.main_input import *
from Sketch_UI.UiPy.precompute_layers_ui import *
from solvers.generation_solver.distance_map import *
from solvers.generation_solver.precompute import *
from solvers.generation_solver.gen_sketch_placement import *
from solvers.generation_solver.adjacency_graph import *
from solvers.generation_solver.new_get_sketch import *
from Sketch_UI.Utils.mesh_utils import *


_CACHE = 0


def store_in_cache(x):
    _CACHE = x


class ParentWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.main_ui = Ui_MainWindow()
        self.main_ui.setupUi(self)


class PrecWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.ui = precPage()
        self.ui.setupUi(self)

    def precompute(self):
        try:
            Precompute().init_by_interface()
        except Exception as e:
            print(e)


class PrecomputeLayersWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.fileName = ""
        self.fileType = ""
        self.ui = PrecumputeLayersUi()
        self.ui.setupUi(self)
        img = QtGui.QPixmap('../resource/icon/upload.jpg')
        self.ui.image_label.setPixmap(img)
        self.ui.image_label.setScaledContents(True)

    def select_graph_file(self):
        graph_relative_path = "/../../solvers/generation_solver/connectivity/"
        graph_file_name, graph_file_type = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件",
                                                                                 os.getcwd() + graph_relative_path,
                                                                   "All Files(*);;jpeg Files(*.jpg);;png Files(*.png)")
        self.graph_file_name = graph_file_name
        self.graph_file_type = graph_file_type
        self.update_graph_combobox()

    def open_image_file(self):
        image_relative_path = "/../../solvers/generation_solver/new_inputs/"
        image_file_name, image_file_type = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件",
                                                                                 os.getcwd() + image_relative_path,
                                                                   "All Files(*);;jpeg Files(*.jpg);;png Files(*.png)")
        self.image_file_name = image_file_name
        self.image_file_type = image_file_type
        img = QtGui.QPixmap(self.image_file_name)
        self.ui.image_label.setPixmap(img)
        self.ui.image_label.setScaledContents(True)
        self.update_image_combobox()

    def update_graph_combobox(self):
        self.ui.graph_select_box.addItem(self.graph_file_name)

    def update_image_combobox(self):
        self.ui.image_select_box.addItem(self.image_file_name)

    def precompute_models(self):
        Precompute().initialize(self.graph_file_name, 1, [self.image_file_name], [1], [], 0, 1, 0, 0)


class DismapWindow(QDialog):
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
        distance_map = DistanceMap(file_path, base_int=24)
        distance_map.generate_distance_map(base_int=24)
        self.hide()

    def update_combobox(self):
        self.ui.comboBox.addItem(self.fileName)


class SupersetWindow(QDialog):
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

    def visualize(self):
        visualize_brick(self.ui.listWidget.currentItem().text())


class AdjacencyGraphWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.fileName = ""
        self.fileType = ""
        self.ui = AdjacencyGraphDialog()
        self.ui.setupUi(self)

    def open_file(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "Select Superset File", os.getcwd(),
                                                                   "LDraw Files(*.ldr)")
        self.fileName = fileName
        self.fileType = fileType
        self.update_combobox()

    def update_combobox(self):
        self.ui.comboBox.addItem(self.fileName)

    def generate_graph(self):
        if self.fileName is not None:
            try:
                gen_adjacency_graph(self.fileName)
            except Exception as e:
                print(e)
        else:
            print("ERR: No selected file!")


class MainInputWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.ui = MainInputDialog()
        self.ui.setupUi(self)
        self.current_file = ""
        self.file_name = []
        self.layer_num = []
        self.counter = 0
        self.layer_count = 4

    def open_file(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "Select Precompute File", os.getcwd(),
                                                                   "Pickle Files(*.pkl)")
        self.current_file = fileName
        self.update_combobox()

    def update_combobox(self):
        self.ui.comboBox.addItem(self.current_file)

    def next(self):
        self.file_name.append(self.current_file)
        self.layer_num.append(self.ui.spinBox.value)
        self.ui.spinBox.setValue(1)
        self.ui.comboBox.clear()
        self.counter += 1
        if self.counter == self.layer_count:
            get_sketch(self.file_name, self.layer_num)
        self.close()


class MainSolverWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.ui = MainSolverDialog()
        self.ui.setupUi(self)
        self.ui.spinBox.setMinimum(1)
        self.ui.spinBox.setValue(1)
        self.layer_count = 1

    def update_layer_count(self):
        self.layer_count = self.ui.spinBox.value()
        print(self.layer_count)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ParentWindow()
    prec = PrecWindow()
    precompute = PrecomputeLayersWindow()
    dismap = DismapWindow()
    superset = SupersetWindow()
    graph = AdjacencyGraphWindow()
    main_solver = MainSolverWindow()
    main_input = MainInputWindow()
    # bind button event
    btn_start = window.main_ui.pushButton_2  # The start button on main window
    btn_start.clicked.connect(main_solver.show)

    main_solver.ui.spinBox.valueChanged.connect(main_solver.update_layer_count)

    btn_start_input = main_solver.ui.pushButton
    btn_start_input.clicked.connect(main_input.show)

    btn_input_tool = main_input.ui.toolButton
    btn_input_tool.clicked.connect(main_input.open_file)

    btn_input_next = main_input.ui.pushButton
    btn_input_next.clicked.connect(main_input.next)

    btn_prec = window.main_ui.pushButton  # The precompute button on main window
    btn_prec.clicked.connect(prec.show)

    btn_precompute = prec.ui.pushButton_2  # The precompute button on precompute window
    # btn_precompute.clicked.connect(prec.precompute)
    btn_precompute.clicked.connect(precompute.show)

    btn_precompute_graph_choose = precompute.ui.open_graph_button
    btn_precompute_graph_choose.clicked.connect(precompute.select_graph_file)

    btn_precompute_image_choose = precompute.ui.open_image_button
    btn_precompute_image_choose.clicked.connect(precompute.open_image_file)

    btn_precompute_model = precompute.ui.push_button
    btn_precompute_model.clicked.connect(precompute.precompute_models)

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
    btn_preview.clicked.connect(superset.visualize)

    btn_generate_superset = superset.ui.pushButton
    btn_generate_superset.clicked.connect(superset.run)

    btn_adjacency = prec.ui.pushButton_4  # Show adjacency graph window.
    btn_adjacency.clicked.connect(graph.show)

    btn_graph_tool = graph.ui.toolButton
    btn_graph_tool.clicked.connect(graph.open_file)

    btn_graph_gen = graph.ui.pushButton
    btn_graph_tool.clicked.connect(graph.generate_graph)

    window.show()
    sys.exit(app.exec())

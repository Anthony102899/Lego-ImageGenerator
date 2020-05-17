# -*- coding: utf-8 -*-
"""
    Animated 3D sinc function
"""

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys, os
from itertools import product
import random

import reader


class Visualizer(object):
    def __init__(self, point_matrices, edges, directions):
        self.app = QtGui.QApplication(sys.argv)

        self.point_matrices = point_matrices
        self.edges = edges
        self.edge_traces = dict()

        self.current_index = 0

        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 40
        self.w.setWindowTitle('Visualizer')
        self.w.setGeometry(0, 110, 1920, 1080)

        # create the background grids
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.translate(-10, 0, 0)
        self.w.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -10, 0)
        self.w.addItem(gy)
        gz = gl.GLGridItem()
        gz.translate(0, 0, -10)
        self.w.addItem(gz)
        self.w.show()



        self.colors = [pg.glColor(i / len(point_matrices) * 254, 254, 254) for i in range(len(point_matrices))]
        self.plot_edges(edges, point_matrices[0], self.colors[0])
        # self.plot_edges(edges, point_matrices[-1], colors[-1])
        self.plot_locus(point_matrices, self.colors)

        base_value = [20, 50, 90, 130, 175, 220, 254]
        random.shuffle(base_value)
        dir_colors = list(product(base_value, base_value, base_value))[::-1]
        init_pts = matrices[0]
        for i, direction in enumerate(directions):
            print(i)
            end_pts = init_pts + direction
            for (s, e) in zip(init_pts, end_pts):
                pts = np.vstack([s, e])
                it = gl.GLLinePlotItem(pos=pts, color=pg.glColor(*dir_colors[i]))
                self.w.addItem(it)


    def plot_edges(self, edges, points, color):
        for i, e in enumerate(edges):
            p1, p2 = points[e[0]], points[e[1]]
            pts = np.vstack([p1, p2])
            self.edge_traces[i] = gl.GLLinePlotItem(pos=pts, color=color, width=1, antialias=True)
            self.w.addItem(self.edge_traces[i])

    def plot_locus(self, point_matrices, colors):
        if len(point_matrices) <= 1:
            return
        for i, locus in enumerate(point_matrices.transpose([1, 0, 2])):
            for j in range(len(locus) - 1):
                seg = locus[j: j + 2,:]
                it = gl.GLLinePlotItem(pos=seg, color=colors[j])
                self.w.addItem(it)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    # def set_plotdata(self, name, points, color, width):
    #     self.traces[name].setData(pos=points, color=color, width=width)

    def update(self):
        self.current_index  = (self.current_index + 1) % len(self.point_matrices)
        points = self.point_matrices[self.current_index]
        for i, e in enumerate(self.edges):
            p1, p2 = points[e[0]], points[e[1]]
            pts = np.vstack([p1, p2])
            self.edge_traces[i].setData(pos=pts, color=self.colors[self.current_index], width=1)


    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)
        self.start()
        self.update()


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    if len(sys.argv) < 2:
        in_file = "data/square_with_parallel_bar.txt.out"
    else:
        in_file = sys.argv[1]
        
    output = in_file[:-7] + "png"
    _, output_filename = os.path.split(output)
    origin_filename = "data/" + output_filename.split('.')[0] + ".txt"
    direction_data_file = in_file[:-3] + "drt.out"
    directions = reader.read_out_data_file(direction_data_file)
    direction_png = output_filename[:-3] + "drt.png"

    edges = reader.read_edge_data_file(origin_filename)
    matrices = reader.read_out_data_file(in_file)

    for i in range(len(directions)):
        for j in range(len(directions[i])):
            norm = np.linalg.norm(directions[i][j])
            if norm > 1e-7:
                directions[i][j] /= np.linalg.norm(directions[i][j])

    v = Visualizer(matrices, edges, directions)
    v.animation()
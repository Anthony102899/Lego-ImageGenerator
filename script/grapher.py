# -*- coding: utf-8 -*-
"""
    Animated 3D sinc function
"""

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys, os

import reader


class Visualizer(object):
    def __init__(self, point_matrices, edges):
        self.app = QtGui.QApplication(sys.argv)

        self.traces = dict()
        self.t = dict()
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


        self.n = 50
        self.m = 1000
        self.y = np.linspace(-10, 10, self.n)
        self.x = np.linspace(-10, 10, self.m)
        self.phase = 0

        point_matrices *= 4

        colors = [pg.glColor(i / len(point_matrices) * 254, 254, 254) for i in range(len(point_matrices))]
        self.plot_edges(edges, point_matrices[0], colors[0])
        self.plot_edges(edges, point_matrices[-1], colors[-1])
        self.plot_locus(point_matrices, colors)

    def plot_edges(self, edges, points, color):
        print(color)
        for e in edges:
            p1, p2 = points[e[0]], points[e[1]]
            pts = np.vstack([p1, p2])
            it = gl.GLLinePlotItem(pos=pts, color=color, width=1, antialias=True)
            self.w.addItem(it)

    def plot_locus(self, point_matrices, colors):
        if len(point_matrices) <= 1:
            return
        for i, locus in enumerate(point_matrices.transpose([1, 0, 2])):
            print(i, locus.shape)
            for j in range(len(locus) - 1):
                seg = locus[j: j + 2,:]
                print(seg.shape)
                it = gl.GLLinePlotItem(pos=seg, color=colors[j])
                self.w.addItem(it)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    # def set_plotdata(self, name, points, color, width):
    #     self.traces[name].setData(pos=points, color=color, width=width)

    # def update(self):
    #     for i in range(self.n):
    #         yi = np.array([self.y[i]] * self.m)
    #         d = np.sqrt(self.x ** 2 + yi ** 2)
    #         z = 10 * np.cos(d + self.phase) / (d + 1)
    #         pts = np.vstack([self.x, yi, z]).transpose()
    #         self.set_plotdata(
    #             name=i, points=pts,
    #             color=pg.glColor((i, self.n * 1.3)),
    #             width=(i + 1) / 10
    #         )
    #         self.phase -= .003

    def animation(self):
        # timer = QtCore.QTimer()
        # timer.timeout.connect(self.update)
        # timer.start(20)
        self.start()
        # self.update()


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    if len(sys.argv) < 2:
        in_file = "data/square_with_parallel_bar.txt.out"
    else:
        in_file = sys.argv[1]
        
    output = in_file[:-7] + "png"
    _, output_filename = os.path.split(output)
    origin_filename = "data/" + output_filename.split('.')[0] + ".txt"

    edges = reader.read_edge_data_file(origin_filename)
    matrices = reader.read_out_data_file(in_file)

    v = Visualizer(matrices, edges)
    v.animation()
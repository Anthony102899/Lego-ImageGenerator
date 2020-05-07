#%%
from matplotlib import pyplot as plt
from itertools import combinations
import numpy as np
import sys
import os

from reader import *

if len(sys.argv) < 2:
    in_file = "data/square_with_parallel_bar.txt.out"
else:
    in_file = sys.argv[1]
    
output = in_file[:-7] + "png"
_, output_filename = os.path.split(output)
origin_filename = "data/" + output_filename.split('.')[0] + ".txt"

to2d = lambda point: (point[0], point[1])

# prepare data
edges = read_edge_data_file(origin_filename)
matrices = read_out_data_file(in_file)
#%%

def plot_points_3d(pts, color):
    for p in pts:
        ax.scatter(p[0], p[1], p[2], c=color)

def plot_points_2d(pts, color):
    x, y = zip(*map(to2d, pts))
    plt.scatter(x, y, c=colors[i])

def plot_edges_3d(pts, color, edges):
    for i, j in edges:
        p, q = pts[i], pts[j]
        x, y, z = zip(p, q)
        plt.plot(x, y, z, c=color)

def plot_edges_2d(pts, color, edges):
    pts = list(map(to2d, pts))
    for i, j in edges:
        p, q = pts[i], pts[j]
        x, y = zip(*map(to2d, (p, q)))
        plt.plot(x, y, c=color)

#%%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = [
    [(i + 1) / len(matrices), 0.1, 0.1]
    for i in range(len(matrices))
]
plot_edges_3d(matrices[0], colors[0], edges)
plot_edges_3d(matrices[-1], colors[-1], edges)
for i, mat in enumerate(matrices):
    # for pt in mat:
    #     ax.scatter(pt[0], pt[1], pt[2])
    plot_points_3d(mat, colors[i])

plt.show()
plt.savefig(output_filename)


# %%


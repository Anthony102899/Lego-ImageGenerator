#%%
from matplotlib import pyplot as plt
from itertools import combinations
import numpy as np
import sys
import os

from reader import *

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

def find_rotation(n_start, n_end):
    v = np.cross(n_start, n_end)
    cosine = np.dot(n_start, n_end)
    sm = np.array(
        [
            [    0, -v[3],  v[2]],
            [ v[3],     0, -v[1]],
            [-v[2],  v[1],     0]
        ]
    )
    rot = np.identity(3) + sm + np.cross(sm, sm) / (1 + cosine)
    return rot


def plot_circle_3d(center, radius, normal):
    thetas = np.linspace(0, 2 * np.pi, 20)
    x = np.cos(thetas)
    y = np.sin(thetas)
    z = 0
    circle = np.hstack([x, y, z])
    # rotation * [0 0 1] = normal
    z_axis = np.array([0, 0, 1])
    rotation = find_rotation(z_axis, normal)
    rotated_circle = np.apply_along_axis(lambda row: np.cross(rotation, row.T).T, 0, circle)
    translated_circle = rotated_circle + center
    x, y, z = zip(*translated_circle)
    plt.plot(x, y, z)
    

#%%
if len(sys.argv) < 2:
    in_file = "data/output/square_with_parallel_bar.txt.out"
else:
    in_file = sys.argv[1]
    
output = in_file[:-7] + "png"
_, output_filename = os.path.split(output)
origin_filename = os.path.join("data", "object", output_filename.split('.')[0] + ".txt") 

to2d = lambda point: (point[0], point[1])

# prepare data
edges = read_data_file(origin_filename, "E")
matrices = read_out_data_file(in_file)
maxval, minval = np.max(matrices), np.min(matrices)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(minval, maxval)
ax.set_ylim(minval, maxval)
ax.set_zlim(minval, maxval)

colors = [
    [(i + 1) / len(matrices), 0.1, 0.1]
    for i in range(len(matrices))
]
plot_edges_3d(matrices[0], colors[0], edges)
plot_edges_3d(matrices[-1], colors[-1], edges)
for i, mat in enumerate(matrices):
    plot_points_3d(mat, colors[i])

plt.savefig(output_filename)

# direction_data_file = in_file[:-3] + "drt.out"
# directions = read_out_data_file(direction_data_file)
# direction_png = output_filename[:-3] + "drt.png"

# plt.clf()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# points = matrices[0]
# plot_edges_3d(points, colors[0], edges)
# for direction in directions:
#     end_points = direction + points
#     plot_points_3d(end_points, (0.5, 0.2, 0.8))
# plt.savefig(direction_png)



# %%


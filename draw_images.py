from matplotlib import pyplot as plt
from itertools import combinations
import numpy as np
import sys
import os

if len(sys.argv) < 2:
    in_file = "square.txt.out"
else:
    in_file = sys.argv[1]
    
output = in_file[:-7] + "png"
_, output_filename = os.path.split(output)
origin_filename = "data/" + output_filename.split('.')[0] + ".txt"

strings_to_ints = lambda it: list(map(float, it))
to2d = lambda point: (point[0], point[1])

def read_edge_data_file(filename):
    with open(filename, "r") as fp:
        lines = [line.strip() for line in fp.readlines()]
    ind = lines.index("E")
    num_edges = int(lines[ind + 1])
    edges = list(map(
        lambda l: list(map(int, l.split(" "))),
        lines[ind + 2: ind + 2 + num_edges]
    ))
    return edges

edges = read_edge_data_file(origin_filename)
matrices = []
with open(in_file) as fp:
    item_num = int(fp.readline().strip())
    for i in range(item_num):
        rows, cols = map(int, fp.readline().strip().split(" "))
        lines = [fp.readline() for _ in range(rows)]
        points = [strings_to_ints(line.strip().split(" ")) for line in lines]
        matrices.append(points)

def print_points(pts, color, edges):
    pts = list(map(to2d, pts))
    for i, j in edges:
        p, q = pts[i], pts[j]
        norm = lambda p, q: sum([(i - j) ** 2 for i, j in zip(p, q)]) ** (0.5)
        print("edge ({}, {}) length {}".format(i, j, norm(p, q)))
        x, y = zip(*map(to2d, (p, q)))
        plt.plot(x, y, c=color)

colors = [
    [(i + 1) / len(matrices), 0.1, 0.1]
    for i in range(len(matrices))
]
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
print_points(matrices[0], colors[0], edges)
print_points(matrices[-1], colors[-1], edges)
for i, mat in enumerate(matrices):
    x, y = zip(*map(to2d, mat))
    plt.scatter(x, y, c=colors[i])

plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.gca().set_aspect('equal', adjustable='box')

plt.savefig(output_filename)

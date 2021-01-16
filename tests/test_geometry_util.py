import numpy as np
from util import geometry_util as gu
from visualization.model_visualizer import visualize_hinges
from testcases import tetra

if __name__ == "__main__":
    pivots = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ]) * 100
    axes = np.array([
        [-0.6002, 0.2442, 0.7616],
        [0.6295, 0.3026, 0.7157],
        [0.2777, -0.2236, 0.9343],
        [0.5090, 0.2976, 0.8077],
    ])
    # axes = np.array([[-0.67047648, 0.73671675, 0.08780502],
    #  [-0.99870806, 0.04578672, 0.02204064],
    #  [0.35756634, -0.71508136, 0.60067042],
    #  [0.38999463, -0.34514895, 0.85368401]])
    model = tetra.square(axes)
    pivots = np.array([j.pivot_point for j in model.joints])
    points = model.point_matrix()
    edges = model.edge_matrix()

    visualize_hinges(points, edges, pivots, axes)
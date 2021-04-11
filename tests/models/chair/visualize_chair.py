import numpy as np
import open3d as o3d
from functools import reduce
from visualization.vismesh import arrow_meshes, visualize_3D
from visualization.model_visualizer import get_geometries_3D, get_mesh_for_points, get_mesh_for_arrows

from model_chair import define

stage = 1
model = define(stage)["model"]

joints = model.joints
# pv, axes = zip(*[(j.pivot, j.rotation_axes[0]) for j in joints if j.rotation_axes is not None])
# trimesh, lines = get_geometries_3D(model.point_matrix(), edges=model.edge_matrix(), show_point=False)
# trimesh += get_mesh_for_arrows(pv, axes, 20)
# o3d.visualization.draw_geometries([trimesh, lines])

data = np.load(f"data/rigid_chair_stage{stage}.npz", allow_pickle=True)
points, edges = data["points"], data["edges"]
eigenvalue, eigenvector = data["eigenvalue"], data["eigenvector"]
stiffness = data["stiffness"]
force = data["force"]

print(eigenvalue)
trimesh, lines = get_geometries_3D(points, edges=edges, show_point=False, show_axis=False)
arrows = get_mesh_for_arrows(points, force.reshape(-1, 3), vec_len_coeff=100)
print(f"sum: {np.sum(force.reshape(-1, 3), axis=0)}")

arrows.paint_uniform_color(np.array([0, 1, 0]))

trimesh += arrows
o3d.visualization.draw_geometries_with_custom_animation(
    [trimesh, lines],
    width=1080,
    height=1080,
    optional_view_trajectory_json_file="view.json")

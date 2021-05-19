import json

from scipy.linalg import null_space
import copy
import numpy as np
from numpy.linalg import matrix_rank, matrix_power, cholesky, inv
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import itertools
from tqdm import tqdm

import util.geometry_util as geo_util
import solvers.rigidity_solver.gradient as gradient
from solvers.rigidity_solver.internal_structure import tetrahedron, triangulation_with_torch
from solvers.rigidity_solver.constraints_3d import select_non_colinear_points
from solvers.rigidity_solver import gradient, algo_core as core, extra_constraint
from solvers.rigidity_solver.eigen_analysis import eigen_analysis
from solvers.rigidity_solver.models import Model, Joint, Beam
from collections import namedtuple

from visualization.model_visualizer import visualize_3D, visualize_2D
from matplotlib import pyplot as plt

with open("boat2.json") as fp:
    data = json.load(fp)

scale = 0.05
density = 0.2
thickness = 2

nodes = {node["id"]: np.array((node["x"], node["y"], node["z"])) for node in data["nodes"]}
all_coord = np.vstack([np.array(n) for n in nodes.values()])
all_coord -= np.mean(all_coord, axis=0)
all_coord *= scale
nodes = {node: all_coord[i] for i, node in enumerate(nodes.keys())}

edges = {edge["id"]: (edge["n1"], edge["n2"]) for edge in data["edges"]}
print("number of beams:", len(edges))
node_edge_map = {
    node_id: tuple([edge_id for edge_id, (p, q) in edges.items() if p == node_id or q == node_id])
    for node_id in nodes
}

model = Model()
beam_map = {
    edge_id: Beam.tetra(
        np.asarray(nodes[p], dtype=np.double), np.asarray(nodes[q], dtype=np.double),
        thickness=thickness,
        density=density
    )
    for edge_id, (p, q) in edges.items()}
model.add_beams(beam_map.values())

for node_id, adj_edges in node_edge_map.items():
    for e_a, e_b in itertools.combinations(adj_edges, 2):
        j = Joint(beam_map[e_a], beam_map[e_b], pivot=nodes[node_id], rotation_axes=np.eye(3))
        model.add_joint(j)

############# MODEL FINISHED

points = model.point_matrix()
edges = model.edge_matrix()

beam_self_rotation = np.zeros((len(model.beams), model.point_count * 3))
for ind, beam in enumerate(model.beams):
    p, q = beam.principle_points
    self_axis = geo_util.normalize(p - q)
    beam_point_indices = model.point_indices()[ind]
    beam_points = points[beam_point_indices]
    disp = geo_util.normalize(np.cross(beam_points - p, self_axis)) / 5
    beam_self_rotation[ind, beam_point_indices[0] * 3: beam_point_indices[-1] * 3 + 3] = disp.reshape(-1)

print("model.point_count", model.point_count)
print(edges.shape)
pairs = model.eigen_solve(
    extra_constr=np.vstack((
        geo_util.trivial_basis(points),
        beam_self_rotation
    )),
    verbose=True)
eigenvalues = np.array([e for e, _ in pairs])


eigind = 0
print("zero-eigval count", np.where(eigenvalues < 1e-12, 1, 0).sum())
print(eigenvalues[eigind:])

plt.show()

for e, v in pairs[eigind:]:
    nodal_disp_map = {}
    for node_id, pos in nodes.items():
        _, selected_indices = select_non_colinear_points(points, num=3, near=pos)
        nodal_disp_map[node_id] = np.mean(v.reshape(-1, 3)[selected_indices], axis=0)

    projection_map = {}
    for na, nb in itertools.combinations(nodal_disp_map, 2):
        diff_disp = nodal_disp_map[na] - nodal_disp_map[nb]
        projection = np.dot(geo_util.normalize(nodes[na] - nodes[nb]), diff_disp)
        projection_map[(na, nb)] = projection

    best_pair = max(projection_map.items(), key=lambda kv: kv[1])
    print(best_pair)
    print(edges)
    reinforcement = Beam.tetra(nodes[best_pair[0][0]], nodes[best_pair[0][1]],
                              thickness=thickness,
                              density=density,)
    model.add_beam(reinforcement)

    with open("boat2-nodes.txt", "w") as fp:
        for node_id, (x, y, z) in nodes.items():
            print(f"{x},{y},{z}", file=fp)

    model.save_json("boat2-reinforced.json")
    model.visualize()

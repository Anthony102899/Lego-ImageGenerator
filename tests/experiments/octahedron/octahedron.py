from solvers.rigidity_solver import (
    algo_core as core,
)
from solvers.rigidity_solver.models import Model, Beam, Joint
from util import geometry_util as geo_util

import torch
import numpy as np
import itertools


t = lambda x, y, z: np.array((x, y, z), dtype=np.double)
nodes = {
    "T": t(11, 11, 26),
    "S": t(11, 11, -24),
    "A": t(1, 1, 1),
    "B": t(21, 1, 1),
    "C": t(21, 21, 1),
    "D": t(1, 21, 1),
}
connectivity = {
    key: (key[0], key[1]) for key in (
        "AT", "BT", "CT", "DT",
        "AS", "BS", "CS", "DS",
        "AB", "BC", "CD", "AD",
    )
}
beam_map = {
    key: Beam.tetra(nodes[p], nodes[q]) for key, (p, q) in connectivity.items()
}

def gen_joint(pvname, conn):
    return [Joint(beam_map.get(conn_pair[0]), beam_map.get(conn_pair[1]), pivot=nodes[pvname], rotation_axes=np.eye(3))
            for conn_pair in itertools.combinations(conn, 2)]
joints = [
    *gen_joint("A", ["AB", "AD", "AS", "AT"]),
    *gen_joint("B", ["AB", "BC", "BS", "BT"]),
    *gen_joint("C", ["BC", "CD", "CS", "CT"]),
    *gen_joint("D", ["AD", "CD", "DS", "DT"]),
    *gen_joint("S", ["AS", "BS", "CS", "DS"]),
    *gen_joint("T", ["AT", "BT", "CT", "DT"]),
]

model = Model()
model.add_beams(beam_map.values())
model.add_joints(joints)

pairs = model.eigen_solve(extra_constr=geo_util.trivial_basis(model.point_matrix()), verbose=True)
eigenvalues = np.array([e for e, _ in pairs])
eigind = 12
assert np.allclose(eigenvalues[:eigind], 0)

smallest_eigenvalue = eigenvalues[eigind]
corresponding_eigenvector = pairs[eigind][1]

from visualization.model_visualizer import get_geometries_3D, get_mesh_for_arrows, colormap, draw_geometries
import open3d as o3d
points, edges = model.point_matrix(), model.edge_matrix()
geometries = get_geometries_3D(points, edges=edges)
arrows = corresponding_eigenvector.reshape(-1, 3)

for part_ind, part_point_indices in enumerate(model.point_indices()):
    arrow_mesh = get_mesh_for_arrows(points[part_point_indices], arrows[part_point_indices],
                                 radius_coeff=0.15, length_coeff=0.3).paint_uniform_color(colormap["orange"])
    o3d.io.write_triangle_mesh(f"output/octahedron-part{part_ind}.obj", arrow_mesh)



import numpy as np
import itertools
from sfepy.discrete import fem

import util.geometry_util as geo_util
import util.meshgen as meshgen
from visualization.model_visualizer import visualize_hinges, visualize_3D
from solvers.rigidity_solver.internal_structure import get_crystal_vertices
from .constraints_3d import select_non_colinear_points, constraints_for_allowed_motions
from .internal_structure import tetrahedral
from .stiffness_matrix import stiffness_matrix_from_mesh


class Model:
    def __init__(self):
        self.beams = []
        self.joints = []

    def point_matrix(self) -> np.ndarray:
        beam_points = np.vstack([b.points for b in self.beams]).reshape(-1, 3)
        # joint_points = np.array([j.virtual_points for j in self.joints]).reshape(-1, 3)
        return np.vstack((
            beam_points,
            # joint_points
        ))

    def edge_matrix(self) -> np.ndarray:
        edge_indices = []
        index_offset = 0
        for beam in self.beams:
            edge_indices.append(beam.edges + index_offset)
            index_offset += beam.point_count
        # for joint in self.joints:
        #     edge_indices.append(joint.edges() + index_offset)
        #     index_offset += joint.virtual_point_count
        matrix = np.vstack([edges for edges in edge_indices if edges.size > 0])
        return matrix

    def constraint_matrix(self) -> np.ndarray:
        matrix = []
        # collect constraints for each joint and stack them
        for joint in self.joints:
            constraints = joint.linear_constraints(self)
            matrix.append(constraints)

        numpy_matrix = np.vstack(matrix) if len(matrix) > 0 else np.empty(0)
        return numpy_matrix

    @property
    def point_count(self):
        return sum(beam.point_count for beam in self.beams)

    def add_beam(self, beam):
        self.beams.append(beam)

    def add_beams(self, beams):
        for beam in beams:
            self.add_beam(beam)

    def add_joint(self, joint):
        self.joints.append(joint)

    def add_joints(self, joints):
        for joint in joints:
            self.add_joint(joint)

    def beam_point_index(self, beam):
        beam_index = self.beams.index(beam)
        return sum(b.point_count for b in self.beams[:beam_index])

    def joint_point_indices(self):
        indices = []
        for joint in self.joints:
            offset_part_1 = self.beam_point_index(joint.part1)
            offset_part_2 = self.beam_point_index(joint.part2)

            indice_on_part_1 = select_non_colinear_points(joint.part1.points, near=joint.pivot_point)[1] + offset_part_1
            indice_on_part_2 = select_non_colinear_points(joint.part2.points, near=joint.pivot_point)[1] + offset_part_2

            indices.append((indice_on_part_1, indice_on_part_2))

        return np.array(indices)


    def visualize(self, arrows=None):
        if arrows is not None:
            visualize_3D(self.point_matrix(), edges=self.edge_matrix(), arrows=arrows.reshape(-1, 3))
        else:
            try:
                pivots, axes = zip(*[(j.pivot, j.rotation_axes[0]) for j in self.joints])
                visualize_hinges(self.point_matrix(), self.edge_matrix(), pivots=pivots, axes=axes)
            except ValueError:
                visualize_3D(self.point_matrix(), edges=self.edge_matrix())


class Beam:
    def __init__(self, points, edges=None):
        if edges is None:
            index_range = range(len(points))
            edges = np.array(list(itertools.combinations(index_range, 2)))

        self._edges = edges
        self.points = points

    @classmethod
    def crystal(cls, p1, p2, crystal_counts):
        orient = (p2 - p1) / np.linalg.norm(p2 - p1)
        crystals = [get_crystal_vertices(c, orient) for c in np.linspace(p1, p2, num=crystal_counts)]
        points = np.vstack(crystals)
        return Beam(points)

    @classmethod
    def tetra(cls, p, q, thickness=1, ori=None):
        points, edges = tetrahedral(p, q, thickness, ori)
        return Beam(points, edges)

    @classmethod
    def dense_tetra(cls, p, q, thickness=1, ori=None):
        points, _ = tetrahedral(p, q, thickness, ori)
        return Beam(points)

    @classmethod
    def vertices(cls, points, orient):
        orient = orient / np.linalg.norm(orient) * 10
        points = np.vstack((points, points + orient))
        return Beam(points)

    @classmethod
    def cube_as_mesh(cls, pivot, u, v, w):
        hashes = hash((tuple(pivot), tuple(u), tuple(v), tuple(w)))
        soup_filename = f"{hashes}.stl"
        mesh_filename = f"{hashes}.mesh"

        meshgen.cube_surface_mesh(soup_filename, pivot, u, v, w)
        meshgen.tetrahedralize(soup_filename, mesh_filename)

        mesh = fem.Mesh.from_file(mesh_filename)
        points = mesh.coors
        nonzero_x, nonzero_y = mesh.create_conn_graph().nonzero()
        edges = np.hstack((nonzero_x.reshape(-1, 1), nonzero_y.reshape(-1, 1)))

        beam = Beam(points, edges)
        beam.stiffness = stiffness_matrix_from_mesh(mesh_filename)
        beam.mesh_filename = mesh_filename
        return beam


    @property
    def edges(self) -> np.ndarray:
        return self._edges

    @property
    def point_count(self):
        return len(self.points)



class Joint:
    def __init__(self, part1, part2, pivot, rotation_axes=None, translation_vectors=None):
        self.part1 = part1
        self.part2 = part2

        self.pivot = np.array(pivot)
        assert self.pivot.shape == (3,)

        if rotation_axes is not None:
            self.rotation_axes = np.array(rotation_axes).reshape(-1, 3)
            assert np.linalg.matrix_rank(self.rotation_axes) == len(self.rotation_axes)
        else:
            self.rotation_axes = None

        if translation_vectors is not None:
            self.translation_vectors = np.array(translation_vectors).reshape(-1, 3)
            assert self.translation_vectors.shape[1] == 3
            assert np.linalg.matrix_rank(self.translation_vectors) == len(self.translation_vectors)
        else:
            self.translation_vectors = None

    def linear_constraints(self, model: Model) -> np.ndarray:
        dim = 3

        constraint_matrices = []
        for source, target in [
            (self.part1, self.part2),
            (self.part2, self.part1)
        ]:
            source_points, source_point_indices = select_non_colinear_points(source.points, near=self.pivot)
            target_points, target_point_indices = select_non_colinear_points(target.points, near=self.pivot)

            source_point_indices += model.beam_point_index(source)
            target_point_indices += model.beam_point_index(target)

            constraints = constraints_for_allowed_motions(
                source_points,
                target_points,
                rotation_pivot=self.pivot,
                rotation_axes=self.rotation_axes,
                translation_vectors=self.translation_vectors,
            )

            print(constraints.shape)

            i, j, k = source_point_indices
            t1, t2, t3 = target_point_indices
            zero_constraint = np.zeros((constraints.shape[0], model.point_count * dim))
            for index, target_index in enumerate(target_point_indices):
                l = target_index
                zero_constraint[:, i * 3: (i + 1) * 3] = constraints[:, 0: 3]
                zero_constraint[:, j * 3: (j + 1) * 3] = constraints[:, 3: 6]
                zero_constraint[:, k * 3: (k + 1) * 3] = constraints[:, 6: 9]
                zero_constraint[:, l * 3: (l + 1) * 3] = constraints[:, (index + 3) * 3: (index + 4) * 3]
                constraint_matrices.append(zero_constraint)

        matrix = np.vstack(constraint_matrices)
        return matrix

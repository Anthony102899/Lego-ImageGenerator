import numpy as np
import itertools
import os

import scipy.linalg
from sfepy.discrete import fem

from .algo_core import generalized_courant_fischer, spring_energy_matrix_accelerate_3D
import util.geometry_util as geo_util
import util.meshgen as meshgen
from util.timer import SimpleTimer
from visualization.model_visualizer import visualize_hinges, visualize_3D
import visualization.model_visualizer as vis
from .constraints_3d import select_non_colinear_points, direction_for_relative_disallowed_motions
from .internal_structure import tetrahedron
from .stiffness_matrix import stiffness_matrix_from_mesh


class Model:
    """
    Represent an assembly
    """
    def __init__(self):
        self.beams = []
        self.joints = []

    def point_matrix(self) -> np.ndarray:
        beam_points = np.vstack([b.points for b in self.beams]).reshape(-1, 3)
        # joint_points = np.array([j.virtual_points for j in self.joints]).reshape(-1, 3)
        return np.vstack((
            beam_points,
        ))

    def point_indices(self):
        beam_point_count = np.array([b.point_count for b in self.beams])
        end_indices = np.cumsum(beam_point_count)
        start_indices = end_indices - beam_point_count
        return [np.arange(start, end) for start, end in zip(start_indices, end_indices)]


    def edge_matrix(self) -> np.ndarray:
        edge_indices = []
        index_offset = 0
        for beam in self.beams:
            edge_indices.append(beam.edges + index_offset)
            index_offset += beam.point_count

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


    def constraints_fixing_first_part(self):
        count = len(self.beams[0].points)
        fixed_coordinates = np.zeros((count * 3, self.point_count * 3))
        for r, c in enumerate(range(count * 3)):
            fixed_coordinates[r, c] = 1

        return fixed_coordinates

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

    def save_json(self, filename: str, **kwargs):
        import json
        from util.json_encoder import ModelEncoder
        with open(filename, "w") as f:
            json.dump(self, f, cls=ModelEncoder, **kwargs)

    def visualize(self, arrows=None, show_axis=True, show_hinge=True, arrow_style=None):
        defaults = {
            "length_coeff": 0.2,
            "radius_coeff": 0.2,
        }
        if arrow_style is not None:
            arrow_style = {
                **defaults,
                **arrow_style,
            }
        else:
            arrow_style = defaults

        geometries = []

        model_mesh = vis.get_lineset_for_edges(self.point_matrix(), self.edge_matrix())
        geometries.append(model_mesh)

        if show_hinge:
            rotation_axes_pairs = [(j.pivot, j.rotation_axes[0]) for j in self.joints if j.rotation_axes is not None]
            if len(rotation_axes_pairs) > 0:
                rotation_pivots, rotation_axes = zip(*rotation_axes_pairs)
                axes_arrows = vis.get_mesh_for_arrows(
                    rotation_pivots,
                    geo_util.normalize(rotation_axes),
                    length_coeff=0.01, radius_coeff=0.4)
                axes_arrows.paint_uniform_color([0.5, 0.2, 0.8])
                geometries.append(axes_arrows)

            translation_vector_pairs = [(j.pivot, j.translation_vectors[0]) for j in self.joints if j.translation_vectors is not None]
            if len(translation_vector_pairs) > 0:
                translation_pivots, translation_vector = zip(*translation_vector_pairs)
                vector_arrows = vis.get_mesh_for_arrows(translation_pivots, translation_vector, length_coeff=0.01, radius_coeff=0.4)
                vector_arrows.paint_uniform_color([0.2, 0.8, 0.5])
                geometries.append(vector_arrows)

            melded_points = [j.pivot for j in self.joints if j.translation_vectors is None and j.rotation_axes is None]
            if len(melded_points) > 0:
                point_meshes = vis.get_mesh_for_points(melded_points)
                geometries.append(point_meshes)

            mesh_frame = vis.o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
            geometries.append(mesh_frame)

        if arrows is not None:
            points = self.point_matrix()
            arrow_mesh = vis.get_mesh_for_arrows(points, arrows.reshape(-1, points.shape[1]), **arrow_style)
            model_meshes = vis.get_geometries_3D(self.point_matrix(), edges=self.edge_matrix(), show_axis=False, show_point=False)
            geometries.extend([arrow_mesh, *model_meshes])

        vis.o3d.visualization.draw_geometries(geometries)


    def joint_stiffness_matrix(self):
        from functools import reduce
        matrix = reduce(lambda x, y: x + y, [j.joint_stiffness(self) for j in self.joints])
        return matrix


    def soft_solve(self, num_pairs=-1, extra_constr=None, verbose=False):
        points = self.point_matrix()
        edges = self.edge_matrix()
        part_stiffness = spring_energy_matrix_accelerate_3D(points, edges, abstract_edges=[])
        joint_stiffness = self.joint_stiffness_matrix()
        K = part_stiffness + joint_stiffness  # global stiffness

        eigenpairs = geo_util.eigen(K, symmetric=True)
        if verbose:
            print(self.report())

        if num_pairs == -1:
            return [(e, v) for e, v in eigenpairs]
        else:
            return [(e, v) for e, v in eigenpairs[:num_pairs]]


    def eigen_solve(self, num_pairs=-1, extra_constr=None, verbose=False):
        points = self.point_matrix()
        edges = self.edge_matrix()

        timer = SimpleTimer()
        stiffness = spring_energy_matrix_accelerate_3D(points, edges, abstract_edges=[])
        timer.checkpoint("K")

        constraints = self.constraint_matrix()
        if extra_constr is not None:
            constraints = np.vstack((constraints, extra_constr))
        K, B = generalized_courant_fischer(stiffness, constraints)
        eigenpairs = geo_util.eigen(K, symmetric=True)
        timer.checkpoint("eig")
        if verbose:
            print(self.report())
            timer.report()

        if num_pairs == -1:
            return [(e, B @ v) for e, v in eigenpairs[:]]
        else:
            return [(e, B @ v) for e, v in eigenpairs[:num_pairs]]

    def __str__(self):
        return str(self.report())

    def report(self) -> dict:
        return {
            **{
                "#parts": len(self.beams),
                "#points": self.point_count,
                "#joints": len(self.joints),
                "#constraints": len(self.constraint_matrix())
            },
            **vars(self)
        }


class Beam:
    def __init__(self, points, edges=None, principle_points=None):
        if edges is None:
            index_range = range(len(points))
            edges = np.array(list(itertools.combinations(index_range, 2)))

        self._edges = edges
        self.points = points
        self.principle_points = principle_points

    @classmethod
    def crystal(cls, p1, p2, crystal_counts):
        from solvers.rigidity_solver.internal_structure import get_crystal_vertices
        orient = (p2 - p1) / np.linalg.norm(p2 - p1)
        crystals = [get_crystal_vertices(c, orient) for c in np.linspace(p1, p2, num=crystal_counts)]
        points = np.vstack(crystals)
        return Beam(points)

    @classmethod
    def tetra(cls, p, q, thickness=1, density=0.333333, ori=None):
        points, edges = tetrahedron(p, q, thickness=thickness, density=density, ori=ori)
        return Beam(points, edges, principle_points=(p, q))

    @classmethod
    def dense_tetra(cls, p, q, density=0.333333, thickness=1, ori=None):
        points, _ = tetrahedron(p, q, density=density, thickness=thickness, ori=ori)
        return Beam(points, principle_points=(p, q))

    @classmethod
    def vertices(cls, points, orient):
        orient = orient / np.linalg.norm(orient) * 10
        points = np.vstack((points, points + orient))
        return Beam(points)

    @classmethod
    def cube_as_mesh(cls, pivot, u, v, w):
        hashes = hash((tuple(pivot), tuple(u), tuple(v), tuple(w)))
        soup_filename = f"data/{hashes}.stl"
        mesh_filename = f"data/{hashes}.mesh"

        import os
        if not os.path.exists(mesh_filename):
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

    @classmethod
    def from_soup_file(cls, soup_filename: str):
        mesh_filename = soup_filename.replace(".obj", ".mesh")
        if not os.path.exists(mesh_filename):
            meshgen.tetrahedralize(soup_filename, mesh_filename)

        beam = cls.from_mesh_file(mesh_filename)
        return beam

    @classmethod
    def from_mesh_file(cls, mesh_filename):
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
    def __init__(self, part1, part2, pivot,
                 rotation_axes=None, translation_vectors=None,
                 soft_translation=None, soft_rotation=None,
                 soft_translation_coeff=None, soft_rotation_coeff=None,
                 ):
        self.part1 = part1
        self.part2 = part2

        self.soft_translation = soft_translation
        self.soft_rotation = soft_rotation
        self.soft_translation_coeff = soft_translation_coeff
        self.soft_rotation_coeff = soft_rotation_coeff

        self.pivot = np.array(pivot)
        assert self.pivot.shape == (3,), f"received pivot {self.pivot}, shape {self.pivot.shape}"

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

        if soft_rotation is not None:
            self.soft_rotation = np.array(soft_rotation).reshape(-1, 3)
            assert np.linalg.matrix_rank(self.soft_rotation) == len(self.soft_rotation)
        else:
            self.soft_rotation = None

        if soft_translation is not None:
            self.soft_translation = np.array(soft_translation).reshape(-1, 3)
            assert self.soft_translation.shape[1] == 3
            assert np.linalg.matrix_rank(self.soft_translation) == len(self.soft_translation)
        else:
            self.soft_translation = None

    def joint_stiffness(self, model: Model) -> np.ndarray:
        dim = 3
        source, target = self.part1, self.part2  # aliases

        source_points, source_point_indices = select_non_colinear_points(source.points, num=3, near=self.pivot)
        target_points, target_point_indices = select_non_colinear_points(target.points, num=3, near=self.pivot)

        source_point_indices += model.beam_point_index(source)
        target_point_indices += model.beam_point_index(target)

        # (n x 18) matrix, standing for prohibitive motion space
        soft_allowed_translation = np.vstack([vectors for vectors in (self.translation_vectors, self.soft_translation) if vectors is not None])
        soft_allowed_rotation = np.vstack([vectors for vectors in (self.rotation_axes, self.soft_rotation) if vectors is not None])

        prohibitive = direction_for_relative_disallowed_motions(
            source_points,
            target_points,
            rotation_pivot=self.pivot,
            rotation_axes=soft_allowed_rotation,
            translation_vectors=soft_allowed_translation,
        )
        prohibitive = geo_util.rowwise_normalize(prohibitive)

        motion_basis = [prohibitive]
        coefficients = [np.ones(prohibitive.shape[0])]
        if self.soft_translation is not None:
            relative_translation = np.vstack([direction_for_relative_disallowed_motions(source_points, target_points, rotation_pivot=self.pivot,
                                                          translation_vectors=scipy.linalg.null_space(translation.reshape(-1, 3)).T,
                                                          rotation_axes=np.eye(3))
                                             for translation in self.soft_translation])
            assert relative_translation.shape == (len(self.soft_translation_coeff), 18), f"number of soft translation ({relative_translation.shape} and coefficient don't match ({len(self.soft_translation_coeff), 18})"
            motion_basis.append(relative_translation)
            coefficients.append(self.soft_translation_coeff)

        if self.soft_rotation is not None:
            relative_rotation = np.vstack([direction_for_relative_disallowed_motions(source_points, target_points, rotation_pivot=self.pivot,
                                                          rotation_axes=scipy.linalg.null_space(rotation.reshape(-1, 3)).T,
                                                          translation_vectors=np.eye(3))
                                          for rotation in self.soft_rotation])

            assert relative_rotation.shape == (len(self.soft_rotation_coeff), 18)
            motion_basis.append(relative_rotation)
            coefficients.append(self.soft_rotation_coeff)

        # cast to numpy array
        motion_basis = np.vstack(motion_basis)
        coefficients = np.concatenate(coefficients)

        # (18 x m) @ (m x m) @ (m x 18) matrix
        local_stiffness = motion_basis.T @ np.diag(coefficients) @ motion_basis
        assert local_stiffness.shape == (18, 18)

        # clip the stiffness matrix to a zero matrix of the same size as the global stiffness matrix
        global_indices = np.concatenate((source_point_indices, target_point_indices))
        stiffness_at_global = np.zeros((model.point_count * dim, model.point_count * dim))
        for local_row_index, global_row_index in enumerate(global_indices):
            for local_col_index, global_col_index in enumerate(global_indices):
                l_row_slice = slice(local_row_index * 3, local_row_index * 3 + 3)
                l_col_slice = slice(local_col_index * 3, local_col_index * 3 + 3)
                g_row_slice = slice(global_row_index * 3, global_row_index * 3 + 3)
                g_col_slice = slice(global_col_index * 3, global_col_index * 3 + 3)
                stiffness_at_global[g_row_slice, g_col_slice] = local_stiffness[l_row_slice, l_col_slice]

        return stiffness_at_global


    def linear_constraints(self, model: Model) -> np.ndarray:
        dim = 3

        constraint_matrices = []
        for source, target in [
            (self.part1, self.part2),
            (self.part2, self.part1)
        ]:
            source_points, source_point_indices = select_non_colinear_points(source.points, num=3, near=self.pivot)
            target_points, target_point_indices = select_non_colinear_points(target.points, num=3, near=self.pivot)

            source_point_indices += model.beam_point_index(source)
            target_point_indices += model.beam_point_index(target)

            constraints = direction_for_relative_disallowed_motions(
                source_points,
                target_points,
                rotation_pivot=self.pivot,
                rotation_axes=self.rotation_axes,
                translation_vectors=self.translation_vectors,
            )

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

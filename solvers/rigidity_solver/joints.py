import numpy as np
import itertools
import util.geometry_util as geo
from solvers.rigidity_solver.internal_structure import get_crystal_vertices
from .constraint_3d import select_points_on_plane


class Model:
    def __init__(self):
        self.beams = []
        self.joints = []

    def point_matrix(self) -> np.ndarray:
        beam_points = np.array([b.points for b in self.beams]).reshape(-1, 3)
        joint_points = np.array([j.virtual_points for j in self.joints]).reshape(-1, 3)
        return np.vstack((beam_points, joint_points))

    def edge_matrix(self) -> np.ndarray:
        edge_indices = []
        index_offset = 0
        for beam in self.beams:
            edge_indices.append(beam.edges() + index_offset)
            index_offset += beam.point_count
        # for joint in self.joints:
        #     edge_indices.append(joint.edges() + index_offset)
        #     index_offset += joint.virtual_point_count
        matrix = np.vstack(edge_indices)
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
        return sum(beam.point_count for beam in self.beams) + sum(joint.virtual_point_count for joint in self.joints)

    def add_beam(self, beam):
        self.beams.append(beam)

    def add_joint(self, joint):
        self.joints.append(joint)

    def beam_point_index(self, beam):
        beam_index = self.beams.index(beam)
        return sum(b.point_count for b in self.beams[:beam_index])

    def joint_point_index(self, joint):
        joint_index = self.joints.index(joint)
        return sum(b.point_count for b in self.beams) + joint_index


class Beam:
    def __init__(self, p1, p2, crystal_counts):
        orient = (p2 - p1) / np.linalg.norm(p2 - p1)
        self.crystals = [get_crystal_vertices(c, orient) for c in np.linspace(p1, p2, num=crystal_counts)]
        self.points = np.vstack(self.crystals)
        # self.points = np.array([p1, p2])

    def edges(self) -> np.ndarray:
        index_range = range(len(self.points))
        pair_indices = np.array(list(itertools.combinations(index_range, 2)))
        return pair_indices

    @property
    def point_count(self):
        return len(self.points)


class Hinge:
    def __init__(self, part1, part2, axis, pivot_point):
        self.part1 = part1
        self.part2 = part2
        self.axis = axis
        self.pivot_point = pivot_point

        self.virtual_points = np.vstack([
            pivot_point,
        ])

    @property
    def virtual_point_count(self) -> int:
        return len(self.virtual_points)

    def edges(self) -> np.ndarray:
        return np.array([[0, 1], [1, 2], [2, 0]])

    def linear_constraints(self, model: Model) -> np.ndarray:
        dim = 3
        delta_pivot_x_ind = model.joint_point_index(self) * 3
        delta_pivot_y_ind = delta_pivot_x_ind + 1
        delta_pivot_z_ind = delta_pivot_x_ind + 2

        constraints = []
        for part in (self.part1, self.part2):
            start_index = model.beam_point_index(part)
            points = part.points

            for i, point in enumerate(points):
                delta_x_ind = (start_index + i) * dim
                delta_y_ind = delta_x_ind + 1
                delta_z_ind = delta_x_ind + 2

                for pivot in self.virtual_points:

                    # point cannot move along beam_vector and the vector perpendicular to it and the axis
                    beam_vector = point - pivot
                    normal_vector = np.cross(self.axis, beam_vector)
                    binormal_vector = np.cross(beam_vector, normal_vector)

                    point_constraints = np.zeros((2, model.point_count * dim))

                    # displacement along beam vector is zero
                    # need a bit of refractoring...
                    point_constraints[0, delta_x_ind] = beam_vector[0]
                    point_constraints[0, delta_y_ind] = beam_vector[1]
                    point_constraints[0, delta_z_ind] = beam_vector[2]
                    point_constraints[0, delta_pivot_x_ind] = -beam_vector[0]
                    point_constraints[0, delta_pivot_y_ind] = -beam_vector[1]
                    point_constraints[0, delta_pivot_z_ind] = -beam_vector[2]

                    # displacement along bi-normal is zero
                    point_constraints[1, delta_x_ind] = binormal_vector[0]
                    point_constraints[1, delta_y_ind] = binormal_vector[1]
                    point_constraints[1, delta_z_ind] = binormal_vector[2]
                    point_constraints[1, delta_pivot_x_ind] = -binormal_vector[0]
                    point_constraints[1, delta_pivot_y_ind] = -binormal_vector[1]
                    point_constraints[1, delta_pivot_z_ind] = -binormal_vector[2]

                constraints.append(point_constraints)


        constraint_matrix = np.vstack(constraints)
        return constraint_matrix

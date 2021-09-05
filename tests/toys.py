import numpy as np
from tests.testsamples.tetra import equilateral_triangle
from tests.testsamples.joint import hinge
from util.geometry_util import trivial_basis
from util import geometry_util
from visualization import model_visualizer
from solvers.rigidity_solver.extra_constraint import z_static
from solvers.rigidity_solver.algo_core import spring_energy_matrix_accelerate_3D
from solvers.rigidity_solver.constraints_3d import select_non_colinear_points, direction_for_relative_disallowed_motions
from scipy.linalg import null_space
import matplotlib.pyplot as plt


def expand_local_to_global(local_matrix, global_shape, global_indices, dim):
    stiffness_at_global = np.zeros(global_shape)
    for local_row_index, global_row_index in enumerate(global_indices):
        for local_col_index, global_col_index in enumerate(global_indices):
            l_row_slice = slice(local_row_index * dim, local_row_index * dim + dim)
            l_col_slice = slice(local_col_index * dim, local_col_index * dim + dim)
            g_row_slice = slice(global_row_index * dim, global_row_index * dim + dim)
            g_col_slice = slice(global_col_index * dim, global_col_index * dim + dim)
            stiffness_at_global[g_row_slice, g_col_slice] = local_matrix[l_row_slice, l_col_slice]

    return stiffness_at_global


def compare_soft_and_hard_constraints():
    axes = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ])

    model = hinge()
    pairs = model.soft_solve()

    points = model.point_matrix() * 20
    edges = model.edge_matrix()
    part_stiffness = spring_energy_matrix_accelerate_3D(points, edges)

    pv = np.array([0, 0, 0])

    joint_stiffness = model.joint_stiffness_matrix()

    sources, source_indices = select_non_colinear_points(model.joints[0].part1.points, 3, near=pv)
    targets, target_indices = select_non_colinear_points(model.joints[0].part2.points, 3, near=pv)
    global_indices = np.concatenate((source_indices, target_indices))

    def compute_ith_eigenpair_with_coeff(i, c):
        global_stiffness = part_stiffness + c * joint_stiffness
        pairs = geometry_util.eigen(global_stiffness, True)
        return pairs[i]

    c_space = np.logspace(-3, 3, 20)
    sixth = list(map(lambda c: compute_ith_eigenpair_with_coeff(6, c), c_space))
    seventh = list(map(lambda c: compute_ith_eigenpair_with_coeff(7, c), c_space))
    plt.plot([e for e, v in sixth], label='6th')
    plt.plot([e for e, v in seventh], label='7th')
    plt.legend()
    plt.show()

    model_visualizer.visualize_3D(points=points, edges=edges, arrows=seventh[0][1].reshape(-1, 3))
    model_visualizer.visualize_3D(points=points, edges=edges, arrows=seventh[-1][1].reshape(-1, 3))
    # pairs = model.eigen_solve(extra_constr=np.vstack([trivial_basis(model.point_matrix(), 3)]))
    # print([e for e, v in pairs])
    # model_visualizer.visualize_3D(points=model.point_matrix(), edges=model.edge_matrix(), arrows=pairs[0][1].reshape(-1, 3))


def compare_motion_basis_2d():
    source_points = np.array([
        [0, 0],
        [1, 0],
    ])
    target_points = np.array([
        [0, 1],
        [0, 2],
    ])

    motion_input = np.array([
        # [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0,  1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, -1.5, 0, 1.5, 0, 0, 0, 0],
    ])
    motion_input = np.array([
        # [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0,  1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 2, 0],
    ])

    part_deformation = np.vstack([
        np.hstack([null_space(geometry_util.trivial_basis(source_points)).T, np.zeros((1, 4))]),
        np.hstack([np.zeros((1, 4)), null_space(geometry_util.trivial_basis(target_points)).T])
    ])

    stiffness_coeff = np.diag([0.1, 0.1, 100])

    rigid = trivial_basis(np.vstack([source_points, target_points]))
    unwanted = np.vstack([part_deformation, rigid])

    motion_basis = np.array([geometry_util.subtract_orthobasis(vec, unwanted) for vec in motion_input])
    motion_basis = geometry_util.orthonormalize(motion_basis)

    joint_stiffness = motion_basis.T @ stiffness_coeff @ motion_basis

    displacement = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    print(displacement.T @ joint_stiffness @ displacement)


def specify_joint_stiffness_partially():
    source_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
    ])
    target_points = np.array([
        [0, 1, 0],
        [0, 2, 0],
        [1, 2, 0],
    ])

    pivot = np.array([0, 0, 0])
    rotation = np.array([[0, 0, 1]])

    soft_translation = np.vstack(([
        np.concatenate(([1, 0, 0, 1, 0, 0, 1, 0, 0], np.zeros((9,)))),
        np.concatenate(([0, 1, 0, 0, 1, 0, 0, 1, 0], np.zeros((9,)))),
        np.concatenate(([0, 0, 1, 0, 0, 1, 0, 0, 1], np.zeros((9,)))),
    ]))

    direction = direction_for_relative_disallowed_motions(
        source_points, target_points,
        rotation_pivot=pivot,
        rotation_axes=rotation,
        translation_vectors=np.eye(3),
    )

    hard_direction = np.array([geometry_util.subtract_orthobasis(v, soft_translation) for v in direction])

    relative_translation_1 = direction_for_relative_disallowed_motions(source_points, target_points, rotation_pivot=pivot, translation_vectors=np.eye(3)[(0, 1), :], rotation_axes=np.eye(3))
    relative_translation_2 = direction_for_relative_disallowed_motions(source_points, target_points, rotation_pivot=pivot, translation_vectors=np.eye(3)[(1, 2), :], rotation_axes=np.eye(3))
    relative_translation_3 = direction_for_relative_disallowed_motions(source_points, target_points, rotation_pivot=pivot, translation_vectors=np.eye(3)[(2, 0), :], rotation_axes=np.eye(3))

    relative_translation = direction_for_relative_disallowed_motions(source_points, target_points, rotation_pivot=pivot, rotation_axes=np.eye(3))
    relative_rotation = direction_for_relative_disallowed_motions(source_points, target_points, rotation_pivot=pivot, translation_vectors=np.eye(3))

    relative_rotation_1 = direction_for_relative_disallowed_motions(source_points, target_points, rotation_pivot=pivot, translation_vectors=np.eye(3), rotation_axes=np.eye(3)[(0, 1), :])
    relative_rotation_2 = direction_for_relative_disallowed_motions(source_points, target_points, rotation_pivot=pivot, translation_vectors=np.eye(3), rotation_axes=np.eye(3)[(1, 2), :])
    relative_rotation_3 = direction_for_relative_disallowed_motions(source_points, target_points, rotation_pivot=pivot, translation_vectors=np.eye(3), rotation_axes=np.eye(3)[(2, 0), :])


    relative_motion_basis = np.vstack([
        relative_rotation_1,
        relative_rotation_2,
        relative_rotation_3,
        relative_translation_1,
        relative_translation_2,
        relative_translation_3,
    ])

    rigid_basis = geometry_util.trivial_basis(np.vstack((source_points, target_points)))

    print(np.linalg.matrix_rank(relative_motion_basis))
    for vec in rigid_basis:
        print(vec @  relative_motion_basis.T)


if __name__ == '__main__':
    specify_joint_stiffness_partially()
    # compare_motion_basis_2d()
    # compare_soft_and_hard_constraints()

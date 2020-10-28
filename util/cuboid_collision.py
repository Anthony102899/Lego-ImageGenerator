import numpy as np
import math

def check_ineq(T, L, cub_vec, ref_vec):
    right = abs(cub_vec[0].dot(L)) + abs(cub_vec[1].dot(L)) + abs(cub_vec[2].dot(L)) + abs(ref_vec[0].dot(L)) + abs(ref_vec[1].dot(L)) + abs(ref_vec[2].dot(L))
    return abs(T.dot(L)) > right

def get_edge_axis(projection_axis, cub_corners_pos):
    local_axis = []
    for i in range(1,4):
        x = cub_corners_pos[0] - cub_corners_pos[i]
        x = x / np.linalg.norm(x)
        projection_axis.append(x)
        local_axis.append(x)
    return projection_axis, local_axis

def cub_collision_detect(cuboid_ref, cuboid):
    center_distance = norm(np.array(cuboid_ref["Origin"]) - np.array(cuboid["Origin"]))
    if center_distance > norm(np.array(cuboid_ref["Dimension"])/2) + norm(np.array(cuboid["Dimension"])/2):
        return False

    corner_transform = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]])

    projection_axis = []
    rotation_ref = cuboid_ref["Rotation"]
    rotation_cub = cuboid["Rotation"]
    center_AB_vec = np.array(cuboid_ref["Origin"]) - np.array(cuboid["Origin"])

    cuboid_dis = np.array([cuboid["Dimension"][0] / 2, cuboid["Dimension"][1] / 2, cuboid["Dimension"][2] / 2])
    cuboid_corner_relative = (np.tile(cuboid_dis, (4, 1))) * corner_transform
    ref_dis = np.array([cuboid_ref["Dimension"][0] / 2, cuboid_ref["Dimension"][1] / 2, cuboid_ref["Dimension"][2] / 2])
    ref_corner_relative = (np.tile(ref_dis, (4, 1))) * corner_transform
    
    ref_corners_pos = np.array(rotation_ref @ ref_corner_relative.transpose()).transpose() + np.array(cuboid_ref["Origin"])
    cub_corners_pos = np.array(rotation_cub @ cuboid_corner_relative.transpose()).transpose() + np.array(cuboid["Origin"])

    projection_axis, A_local_axis = get_edge_axis(projection_axis, cub_corners_pos)
    projection_axis, B_local_axis = get_edge_axis(projection_axis, ref_corners_pos)

    cub_vec = A_local_axis * np.array([cuboid_dis]).transpose()
    ref_vec = B_local_axis * np.array([ref_dis]).transpose()

    for axis in projection_axis:
        if check_ineq(center_AB_vec, axis, cub_vec, ref_vec):
            return False

    for Aedge in A_local_axis:
        for Bedge in B_local_axis:
            cross = np.cross(Aedge, Bedge)
            if check_ineq(center_AB_vec, cross, cub_vec, ref_vec):
                return False
    return True

def main():
    cuboid_1 = {"Origin": [1.3033025, -9.3033025,5.656856], "Rotation": [[ 0.5, -0.707107, -0.5],[ 0.5,0.707107, -0.5],[ 0.707107,0 ,0.707107]], "Dimension": [18, 11, 2]}
    cuboid_2 = {"Origin": [-1.414214, 1.414214, 0], "Rotation": [[ 0.5, -0.707107, -0.5],[ 0.5,0.707107, -0.5],[ 0.707107,0 ,0.707107]], "Dimension": [18, 2, 18]}
    print(cub_collision_detect(cuboid_1,cuboid_2))

if __name__ == '__main__':
    main()
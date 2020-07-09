import numpy as np
import math

# the cubes do not collide with each other in one axis only when the max of one cube is smaller than another cube
# under the same axis
def collide_check(ref_min, ref_max, minn, maxx):
    if ref_min > maxx or ref_max < minn:
        return False
    return True

# Detect collision in each dimension
def collision_detect(ref_corner, cuboid_corner):
    x_min = np.min(cuboid_corner[:, 0])
    x_max = np.max(cuboid_corner[:, 0])
    y_min = np.min(cuboid_corner[:, 1])
    y_max = np.max(cuboid_corner[:, 1])
    z_min = np.min(cuboid_corner[:, 2])
    z_max = np.max(cuboid_corner[:, 2])

    xref_min = np.min(ref_corner[:, 0])
    xref_max = np.max(ref_corner[:, 0])
    yref_min = np.min(ref_corner[:, 1])
    yref_max = np.max(ref_corner[:, 1])
    zref_min = np.min(ref_corner[:, 2])
    zref_max = np.max(ref_corner[:, 2])

    x_collide = collide_check(xref_min, xref_max, x_min, x_max)
    y_collide = collide_check(yref_min, yref_max, y_min, y_max)
    z_collide = collide_check(zref_min, zref_max, z_min, z_max)

    return (x_collide and y_collide and z_collide)

def cub_collision_detect(cuboid_ref, cuboid):
    corner_transform = np.array([[1,1,1], [1,-1,1], [-1,-1,1], [-1,1,1], [1,1,-1], [1,-1,-1], [-1,-1,-1], [-1,1,-1]])
    projection_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # information of the projection axis

    # Calculate all possible projection matrices for both cubes in respect to the base frame
    projection_axis = []
    rotation_ref = cuboid_ref["Rotation"]
    rotation_cub = cuboid["Rotation"]
    axis_ref = projection_matrix @ rotation_ref
    axis_cub = projection_matrix @ rotation_cub
    # Projections that reverse rotation of ref and cub
    projection_axis.append(axis_ref)
    projection_axis.append(axis_cub)
    # Projection matrices relative to three faces of cuboid_ref
    for i in range(3):
        base_axis = axis_ref[:,i].reshape(3)
        proj_mat = np.zeros((3, 3))
        for j in range(3):
            cross = np.cross(base_axis, axis_cub[:,j].reshape(3))
            proj_mat[:,j] = cross.reshape(3)
        projection_axis.append(proj_mat)

    # Position of each corner point relative to cube's center (do not consider the position relative to base frame's origin)
    cuboid_center = np.array([cuboid["Dimension"][0] / 2, cuboid["Dimension"][1] / 2, cuboid["Dimension"][2] / 2])
    cuboid_corner_relative = (np.tile(cuboid_center, (8, 1))) * corner_transform
    # Position of each corner point relative to cube's center (do not consider the position relative to base frame's origin)
    ref_center = np.array([cuboid_ref["Dimension"][0] / 2, cuboid_ref["Dimension"][1] / 2, cuboid_ref["Dimension"][2] / 2])
    ref_corner_relative = (np.tile(ref_center, (8, 1))) * corner_transform
    # Add origin & rotation to get the absolute coordinates of each corner point
    ref_corners_pos = ref_corner_relative @ rotation_ref + np.array(cuboid_ref["Origin"])
    cub_corners_pos = cuboid_corner_relative @ rotation_cub + np.array(cuboid["Origin"])

    for axis in projection_axis:
        cub_corners_proj = cub_corners_pos @ axis.T
        ref_corners_proj = ref_corners_pos @ axis.T
        collision_decision = collision_detect(ref_corners_proj, cub_corners_proj)
        if collision_decision == False:
            break
    return collision_decision

def main():
    cuboid_1 = {"Origin": [0, 0, 0], "Rotation": [[1, 0, 0], [0, 0, -1], [0, 1, 0]], "Dimension": [3, 1, 2]}
    cuboid_2 = {"Origin": [-0.8, 0, -0.5], "Rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "Dimension": [1, 0.5, 0.5]}
    print(cub_collision_detect(cuboid_1,cuboid_2))

if __name__ == '__main__':
    main()
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

def Check_Collision(cuboid_ref, cuboid):
    T_matrix = np.array([[1,1,1], [1,-1,1], [-1,-1,1], [-1,1,1], [1,1,-1], [1,-1,-1], [-1,-1,-1], [-1,1,-1]])
    projection_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Here stores the information of the projection axis

    # Calculate all possible projection axis for both cubes in respect to the base frame
    projection_axis = []
    rotation_ref = cuboid_ref["Rotation"]
    rotation_cub = cuboid["Rotation"]
    axis_ref = projection_matrix @ rotation_ref
    axis_cub = projection_matrix @ rotation_cub
    projection_axis.append(axis_ref)
    projection_axis.append(axis_cub)
    for i in range(3):
        base_axis = axis_ref[:,i].reshape(3)
        PA = np.zeros((3, 3))
        for j in range(3):
            cross = np.cross(base_axis, axis_cub[:,j].reshape(3))
            PA[:,j] = cross.reshape(3)
        projection_axis.append(PA)

    # Rotate each corner point relative to cube's center (do not consider the position relative to base frame's origin)
    cuboid_corner_initial = np.array([cuboid["Dimension"][0] / 2, cuboid["Dimension"][1] / 2, cuboid["Dimension"][2] / 2])
    cuboid_corner = np.tile(cuboid_corner_initial, (8, 1))
    cuboid_corner = cuboid_corner * T_matrix

    # Position of each corner point relative to cube's center (do not consider the position relative to base frame's origin)
    ref_corner_initial = np.array([cuboid_ref["Dimension"][0] / 2, cuboid_ref["Dimension"][1] / 2, cuboid_ref["Dimension"][2] / 2])
    ref_corner = np.tile(ref_corner_initial, (8, 1))
    ref_corner = ref_corner * T_matrix
    # Add origin to get the absolute coordinates of each corner point
    ref_corners_pos = ref_corner @ rotation_ref + np.array(cuboid_ref["Origin"])
    cub_corners_pos = cuboid_corner @ rotation_cub + np.array(cuboid["Origin"])

    collision_or_not = True
    for axis in projection_axis:
        cuboid_corner_proj = cub_corners_pos @ axis.T
        ref_corner_proj = ref_corners_pos @ axis.T
        collision_decision = collision_detect(ref_corner_proj, cuboid_corner_proj)
        collision_or_not = collision_decision and collision_or_not
        if collision_or_not == False:
            break
    return collision_or_not

def cub_collision_detect(cuboid_1, cuboid_2):
    result = Check_Collision(cuboid_1, cuboid_2)   #  In reference of cuboid1
    return result

def main():
    cuboid_1 = {"Origin": [0, 0, 0], "Orientation": [0, 0, 0], "Dimension": [3, 1, 2]}
    cuboid_2 = {"Origin": [-0.8, 0, -0.5], "Orientation": [0, 0, 0.2], "Dimension": [1, 0.5, 0.5]}
    print(cub_collision_detect(cuboid_1,cuboid_2))

if __name__ == '__main__':
    main()
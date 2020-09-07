import os
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.connections.conn_type import compute_conn_type
import util.cuboid_geometry as cu_geo
import numpy as np
import itertools as iter

collider_path = "/Applications/Studio 2.0/ldraw/collider"
connectivity_path = "/Applications/Studio 2.0/ldraw/connectivity"

def get_bbox(brick):
    bbox = []
    brick_id = brick.template.id
    brick_rot = brick.get_rotation()
    brick_trans = brick.get_translation()
    for line in open(os.path.join(collider_path, f"{brick_id}.col")):
        line = (line.split(" "))[:17]
        line = [float(x) for x in line]
        init_orient = (np.array(line[2:11])).reshape((3,3))
        #print("init_orient =\n", init_orient)
        init_origin = np.array(line[11:14])
        #print("init_origin = ", init_origin)
        init_dim = init_orient @ np.array(line[14:17])  # in (x,y,z) format
        #print("init_size = ", init_dim)

        origin = brick_rot @ init_origin + brick_trans
        #print("\norigin =\n", origin)
        rotation = brick_rot @ init_orient
        #print("rotation =\n", rotation)
        bbox.append({"Origin": origin, "Rotation": rotation, "Dimension": init_dim})
    return bbox

def get_cpoint(brick):
    cps = []
    brick_id = brick.template.id
    brick_rot = brick.get_rotation()
    brick_trans = brick.get_translation()
    binFile = open(os.path.join(connectivity_path, f"{brick_id}.conn"),'rb')
    
    data = []
    """ """
    for line in binFile:
        print(len(line))
        print([line[i] for i in range(len(line))])

if __name__ == "__main__":
    bricks = (read_bricks_from_file("./debug/3005.ldr"))
    #get_cpoint(bricks[0])
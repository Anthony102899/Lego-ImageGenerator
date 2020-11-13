import numpy as np
import os
import math
import cv2
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.bricks.brickinstance import BrickInstance, get_corner_pos
from solvers.generation_solver.crop_model import RGB_to_Hex
from solvers.generation_solver.draw_bbox import write_bricks_w_bbox
from multiprocessing import Pool
from functools import partial

# return a polygon obj 
def proj_bbox(brick:BrickInstance): 
    bbox_corner = np.array(get_corner_pos(brick, four_point=True))
    bbox_corner = [[coord[0], coord[2]] for coord in bbox_corner]
    polygon_ls = []
    for i in range(0, len(bbox_corner), 4):
        polygon = Polygon(bbox_corner[i:i+4])
        polygon_ls.append(polygon)
    polygon = unary_union(polygon_ls)
    return polygon

# return a vector of sd
def color_sd_av(img, brick, offset_i, offset_j):
    polygon = proj_bbox(brick)
    mini, minj, maxi, maxj = polygon.bounds
    rgbs = []
    for x in range(math.floor(mini), math.ceil(maxi)+ 1):
        for y in range(math.floor(minj), math.ceil(maxj) + 1):
            point = Point(x, y)
            if polygon.contains(point):
                try:
                    rgbs.append((img[y - offset_j, x - offset_i])[::-1])
                except:
                    continue
    return np.std(rgbs, axis = 0), np.average(rgbs, axis = 0)

# get a new brick with the nearest color
def check_sketch(brick, img, minx, minz):
    sd, color = color_sd_av(img, brick, minx, minz) # the nearest color
    nearby_hex = RGB_to_Hex(color)
    new_brick = BrickInstance(brick.template, brick.trans_matrix, nearby_hex)
    return new_brick

def get_sketch(img, plate_set, minx, minz):
    with Pool(20) as p:
        result_crop = p.map(partial(check_sketch, img=img, minx=int(minx), minz=int(minz)), plate_set)
    return result_crop

if __name__ == "__main__":
    img_path = os.path.join(os.path.dirname(__file__), "super_graph/Google-Photos.JPG")
    img = cv2.imread(img_path)
    plate_path = "super_graph/for sketch/" + input("Enter file name in sketch folder: ")
    plate_path = os.path.join(os.path.dirname(__file__), plate_path)
    plate_set = read_bricks_from_file(plate_path)
    
    base_count = 0
    for i in range(10):
        if plate_set[i].color == 15:
            base_count += 1
        else:
            break
    plate_base = plate_set[:base_count]
    plate_set = plate_set[base_count:]

    centers = np.array([brick.get_translation() for brick in plate_set])
    maxx, minx = np.amax(centers[:,0]), np.amin(centers[:,0])
    maxz, minz = np.amax(centers[:,2]), np.amin(centers[:,2])
    print("#bricks in plate: ", len(plate_set))

    _, filename = os.path.split(img_path)
    filename = (filename.split("."))[0]
    _, platename = os.path.split(plate_path)
    platename = ((platename.split("."))[0]).split(" ")
    basename = int(platename[1].split("=")[1]) # a number indicating shape of base
    platename = platename[0]

    # resize image to fit the brick
    img = cv2.resize(img, (int(maxx - minx) + 1, int(maxz - minz) + 1))

    result = get_sketch(img, plate_set, minx, minz) + plate_base
    debugger = MyDebugger("sketch")
    write_bricks_to_file(result, file_path=debugger.file_path(f"{filename} b={base_count} n={len(result)} {platename}.ldr"))
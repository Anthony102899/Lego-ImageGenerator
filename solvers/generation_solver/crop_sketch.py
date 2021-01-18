import numpy as np
import os
import math
import cv2
import pickle5 as pickle
from shapely.geometry import Polygon, Point
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.bricks.brickinstance import BrickInstance, get_corner_pos
import solvers.generation_solver.sketch_util as util
from multiprocessing import Pool
from functools import partial

# return color or sd or -1
def crop_ls(rgbs, sd):
    if len(rgbs) == 0:
        if sd:
            return -1
        return []
    if sd:
        return float(round(np.sum(np.std(rgbs, axis = 0)), 4) + 0.0001)
    return np.average(rgbs, axis = 0)

# return *result_sd* and *result_color*
def ls_from_layout(img, plate_set, base_int):
    with Pool(20) as p:
        rgbs_ls = p.map(partial(util.get_cover_rgb, img=img, base_int=base_int), plate_set)
        result_sd = p.map(partial(crop_ls, sd=True), rgbs_ls)
        result_color = p.map(partial(crop_ls, sd=False), rgbs_ls)
    return result_sd, result_color

if __name__ == "__main__":
    img_path = os.path.join(os.path.dirname(__file__), "super_graph/images/Google-Photos_blue.png")
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    degree = int(input("Enter rotation angle: "))
    scale = int(input("Enter scalling factor: "))
    img = util.center_crop(img, scale)
    img = util.rotate_image(img, degree)

    plate_path = "super_graph/for sketch/" + "['49668', '27263', '27925', '3024', '3023', '3710', '24299', '24307', '43722', '43723'] base=24 n=9629 r=1.ldr"
    plate_path = os.path.join(os.path.dirname(__file__), plate_path)
    plate_set = read_bricks_from_file(plate_path)
    base_count = util.count_base(plate_set)
    plate_base = plate_set[:base_count]
    plate_set = plate_set[base_count:]
    print("#bricks in plate: ", len(plate_set))

    _, filename = os.path.split(img_path)
    filename = (filename.split("."))[0]
    if not scale == 1 and degree == 0:
        cv2.imwrite(os.path.join(os.path.dirname(__file__), f"super_graph/images/{filename}_{degree}_{scale}.png"), img)
    _, platename = os.path.split(plate_path)
    platename = (((platename.split("."))[0]).split("n="))[0]

    cpoints = np.array([len(base.get_current_conn_points()) / 2 for base in plate_base])
    base_int = int(math.sqrt(np.sum(cpoints)))

    # resize image to fit the brick
    img = cv2.resize(img, (base_int * 20 + 1, base_int * 20 + 1))

    result_sd, result_color = ls_from_layout(img, plate_set, base_int)
    result_sd = [0.0001 for i in range(base_count)] + result_sd
    result_color = [i for i in result_color if len(i) == 3]
    result_color = np.average(result_color, axis = 0)

    crop_result = util.Crop(result_sd, result_color, base_count, filename + f"_{degree}_{scale}", platename)
    pickle.dump(crop_result, open(os.path.join(os.path.dirname(__file__), f"connectivity/crop {filename}_{degree}_{scale} b={base_count} {platename}.pkl"), "wb"))
    print(f"Saved at {filename}_{degree}_{scale}")
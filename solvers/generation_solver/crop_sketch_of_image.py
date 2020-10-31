import numpy as np
import time
import os
import itertools as iter
from PIL import Image 
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.bricks.brickinstance import BrickInstance, get_corner_pos
from multiprocessing import Pool
from functools import partial
from solvers.generation_solver.crop_model import RGB_to_Hex

# get a new brick with the nearest color
def check_sketch(brick, img, minx, minz):
    center = brick.get_translation()
    cordinate = int(round(center[0] - minx)), int(round(center[2] - minz))
    color = img.getpixel(cordinate) # the nearest color
    nearby_hex = RGB_to_Hex(color)
    new_brick = BrickInstance(brick.template, brick.trans_matrix, nearby_hex)
    return new_brick

def get_sketch(img, plate_set, minx, minz):
    #colors_rgb = list(img.getdata())
    with Pool(20) as p:
        result_crop = p.map(partial(check_sketch, img=img, minx=minx, minz=minz), plate_set)
    return result_crop

if __name__ == "__main__":
    img_path = os.path.join(os.path.dirname(__file__), "super_graph/google.png")
    img = Image.open(img_path).convert("RGB")
    plate_path = os.path.join(os.path.dirname(__file__), "super_graph/for sketch/['3024'] base=12 n=146 r=1.ldr")
    plate_set = read_bricks_from_file(plate_path)
    plate_base = plate_set[:2]
    plate_set = plate_set[2:]
    centers = np.array([brick.get_translation() for brick in plate_set])
    maxx, minx = np.amax(centers[:,0]), np.amin(centers[:,0])
    maxz, minz = np.amax(centers[:,2]), np.amin(centers[:,2])
    print("#bricks in plate: ", len(plate_set))

    _, filename = os.path.split(img_path)
    filename = (filename.split("."))[0]
    _, platename = os.path.split(plate_path)
    platename = ((platename.split("."))[0]).split(" ")
    basename = int(platename[1].split("=")[1])
    platename = platename[0]

    # resize image to fit the brick
    img = img.resize((int(maxx - minx) + 1, int(maxz - minz) + 1))

    start_time = time.time()
    result = get_sketch(img, plate_set, minx, minz) + plate_base
    end_time = time.time()

    debugger = MyDebugger("sketch")
    write_bricks_to_file(result, file_path=debugger.file_path(f"{filename} n={len(result)} {platename} t={round(end_time - start_time, 2)}.ldr"))
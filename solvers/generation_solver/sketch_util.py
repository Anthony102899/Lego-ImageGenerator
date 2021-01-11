import numpy as np
import math
import json
from shapely.geometry import Polygon, Point
from bricks_modeling.bricks.brickinstance import BrickInstance, get_corner_pos
from shapely.ops import unary_union

class Crop:            # sd of nodes
    def __init__(self, result_sd, result_color, base_count, filename, platename):
        self.result_sd = result_sd
        self.result_color = result_color
        self.base_count = base_count
        self.filename = filename
        self.platename = platename

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

def hex_to_rgb(hexx):
    value = hexx.lstrip('0x2')
    lv = len(value)
    if lv == 0:
        return np.array([0,0,0])
    rgb = [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]
    return np.array(rgb)

def RGB_to_Hex(rgb):
    color = '0x2'
    for i in rgb[:3]:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

# get a new brick with the input color
def color_brick(brick, rgb_color):
    color_hex = RGB_to_Hex(rgb_color)
    new_brick = BrickInstance(brick.template, brick.trans_matrix, color_hex)
    return new_brick

# return a list of rgb colors covered by brick *rgbs*
def get_cover_rgb(brick, img, base_int):
    polygon = proj_bbox(brick)
    mini, minj, maxi, maxj = polygon.bounds
    rgbs = []
    channel = len(img[0][0])
    for x in range(math.floor(mini), math.ceil(maxi) + 1):
        for y in range(math.floor(minj), math.ceil(maxj) + 1):
            if x < 0 or y < 0 or x > base_int * 20 or y > base_int * 20:
                return []
            point = Point(x, y)
            if polygon.contains(point):
                try:
                    rgb_color = (img[y, x][:3])[::-1]
                    if channel == 4 and (img[y, x][0] == 0 or sum(np.array(rgb_color)) <= 10):
                        return []
                    # not transparent
                    else:
                        rgbs.append(rgb_color)
                except:
                    continue
    return rgbs

# return difference between input and brickset (solution) (without base)
def calculate_v(brick_set,img, base_int, polygon_ls):
    dif_sum = [0, 0, 0]
    for i in range(len(brick_set)):
        brick = brick_set[i]
        polygon = polygon_ls[i]
        brick_color = hex_to_rgb(brick.color)
        mini, minj, maxi, maxj = polygon.bounds
        dif = [0, 0, 0]
        for x in range(math.floor(mini), math.ceil(maxi) + 1):
            for y in range(math.floor(minj), math.ceil(maxj) + 1):
                point = Point(x, y)
                if polygon.contains(point):
                    try:
                        img_color = (img[y, x])[::-1]
                        dif = [dif[i] + abs(brick_color[i] - img_color[i]) * 1e-6 for i in range(3)]
                    except:
                        continue
        dif_sum = [round(dif_sum[i] + dif[i], 3) for i in range(3)]
    return - np.sum(dif_sum)

# *brick* is the lower one
def calculate_overlap_v(brick, brick2, img, base_int):
    polygon1 = proj_bbox(brick)
    polygon2 = proj_bbox(brick2)
    dif_polygon = polygon1.difference(polygon2)
    return calculate_v([brick], img, base_int, [dif_polygon])

# return an integer in [0,1]
def cal_border(brickset, base_int):
    standard = base_int * 4 - 4
    maxx = base_int * 20 - 10
    count = 0
    for brick in brickset:
        cpoints = brick.get_current_conn_points()
        cpoints_pos = [[cp.pos[0], cp.pos[2]] for cp in cpoints]
        for z in range(10, base_int * 20 -9, 10):
            if [10, z] in cpoints_pos or [maxx, z] in cpoints_pos:
                count += 1
            if z < 20 or z > maxx - 10:
                continue
            if [z, 10] in cpoints_pos or [z, maxx] in cpoints_pos:
                count += 1
    return count / standard

def load_data(brick_database=["regular_plate.json"]):
    data = []
    for data_base in brick_database:
        database_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "bricks_modeling", "database", data_base)
        with open(database_file) as f:
            temp = json.load(f)
            data.extend(temp)
    return data

def get_weight():
    data = load_data()
    area = {}
    for brick in data:
        if "weight" in brick.keys():
            area.update({brick["id"]: brick["weight"]})
        else:
            area.update({brick["id"]: 1})
    return area

def get_area():
    data = load_data()
    area = {}
    for brick in data:
        if "area" in brick.keys():
            area.update({brick["id"]: brick["area"]})
    return area

if __name__ == "__main__":
    print(RGB_to_Hex([20, 153, 233]))
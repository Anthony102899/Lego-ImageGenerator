import json
from os import path

from bricks_modeling.bricks.brickinstance import BrickInstance
from bricks_modeling.bricks.bricktemplate import BrickTemplate
from bricks_modeling.connections.connpoint import CPoint
from bricks_modeling.connections.connpointtype import stringToType
from util.debugger import MyDebugger
from util.geometry_util import gen_lateral_vec
import numpy as np


def get_all_brick_templates(
    brick_database=[
        "technic_beam.json",
        "technic_axle.json",
        "technic_pin.json",
        "technic_connector.json",
        "regular_cuboid.json",
        "regular_plate.json",
        "regular_slope.json",
        "regular_other.json",
        "regular_circular.json"
    ]
):
    data = []
    for data_base in brick_database:
        database_file = path.join(
            path.dirname(path.dirname(__file__)), "database", data_base
        )
        with open(database_file) as f:
            temp = json.load(f)
            data.extend(temp)
            
    brick_templates = []
    template_ids = []
    for brick in data:
        cpoints = []
        for connPoint in brick["connPoint"]:
            cpoints.append(
                CPoint(
                    pos=connPoint["pos"],
                    orient=connPoint["orient"],
                    bi_orient=gen_lateral_vec(np.array(connPoint["orient"])),
                    type=stringToType[connPoint["type"]],
                )
            )
        brick_template = BrickTemplate(cpoints, brick["id"])
        brick_templates.append(brick_template)
        template_ids.append(brick["id"])

    return brick_templates, template_ids


if __name__ == "__main__":
    debugger = MyDebugger("test")

    brick_templates, _ = get_all_brick_templates()

    brick = BrickInstance(brick_templates[0])
    print(brick.to_ldraw())
    brick.to_file(debugger.file_path("file.ldr"))
    input("")

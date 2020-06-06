import numpy
import json
from bricks.BrickTemplate import BrickTemplate
from bricks.ConnPoint import CPoint
from bricks.BrickInstance import BrickInstance
from util.debugger import MyDebugger

def get_all_brick_templates():
    with open('./database/brick_database.json') as f:
        data = json.load(f)

    brick_templates = []
    template_ids  = []
    for brick in data:
        cpoints = []
        for connPoint in brick["connPoint"]:
            cpoints.append(CPoint(pos = connPoint["pos"], orient = connPoint["orient"], type = connPoint["type"]))
        brick_template = BrickTemplate(cpoints, brick["id"])
        brick_templates.append(brick_template)
        template_ids.append(brick["id"])

    return brick_templates, template_ids

if __name__ == "__main__":
    debugger = MyDebugger("test")

    brick_templates = get_all_brick_templates()

    brick = BrickInstance(brick_templates[0])
    print(brick.to_ldraw())
    brick.to_file(debugger.file_path("file.ldr"))
    input("")
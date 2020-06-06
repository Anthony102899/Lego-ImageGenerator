from bricks.BrickInstance import BrickInstance
from Abstract.Abstract_parts import Points, Edges, Anchors, Pins
from typing import List

def write_bricks_to_file(bricks: List[BrickInstance], file_path):
    file = open(file_path, "a")
    ldr_content = "\n".join([brick.to_ldraw() for brick in bricks])
    file.write(ldr_content)
    file.close()

def write_features_to_input(points:Points, edges:Edges, anchors:Anchors, pins:Pins, file_path):
    file = open(file_path, "a")
    file.write("P\n")
    file.write(str(len(points.points)))
    file.write("\n")
    for i in range(len(points.points)):
        file.write(" ".join((str((points.points[i])[0][0]), str((points.points[i])[0][1]),str((points.points[i])[0][2]))))
        file.write("\n")
    file.write("E\n")
    file.write(str(len(edges.edges)))
    file.write("\n")
    for i in range(len(edges.edges)):
        file.write(" ".join((str((edges.edges[i])[0]), str((edges.edges[i])[1]))))
        file.write("\n")
    file.write("anchors\n")
    file.write(str(len(anchors.anchors)))
    file.write("\n")
    for i in range(len(anchors.anchors)):
        file.write(" ".join((str((anchors.anchors[i])[0]), str((anchors.anchors[i])[1]), str((anchors.anchors[i])[2]))))
        file.write("\n")
    file.write("pins\n")
    file.write(str(len(pins.pins)))
    file.write("\n")
    for i in range(len(pins.pins)):
        file.write(" ".join((str((pins.pins[i])[0]), str((pins.pins[i])[1]), str((pins.pins[i])[2]))))
        file.write("\n")
import math
import numpy as np
from bricks_modeling.bricks.brickinstance import BrickInstance
from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.structure_graph import ConstructionGraph
from util.debugger import MyDebugger


if __name__ == "__main__":
    debugger = MyDebugger("test")
    bricks = read_bricks_from_file("./data/LEGO_models/full_models/cube7.ldr")
    structure_graph = ConstructionGraph(bricks)
    print(structure_graph.to_json())
    structure_graph.show()

from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from util.geometry_util import *
from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.bricks.brickinstance import BrickInstance
from bricks_modeling.bricks.bricktemplate import BrickTemplate
from bricks_modeling.connections.connpoint import CPoint
from bricks_modeling.connections.connpointtype import ConnPointType
import numpy as np


if __name__ == "__main__":
    debugger = MyDebugger("test")
    brick_templates, template_ids = get_all_brick_templates()

    for template in brick_templates:
        brickInstance = BrickInstance(template, np.identity(4, dtype=float), 15)
        write_bricks_to_file(
            [brickInstance],
            file_path=debugger.file_path(f"{template.id}_test.ldr"),
            debug=True,
        )

    matrix = np.array([
        [0, 1, 0],
        [1, 2, 0],
        [0, 0, 3]
    ], dtype=np.float64)
    print(eigen(matrix, symmetric=True))

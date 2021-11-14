from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.bricks.brickinstance import BrickInstance
import numpy as np
from util.geometry_util import get_random_transformation


if __name__ == "__main__":
    debugger = MyDebugger("test")
    brick_templates, template_ids = get_all_brick_templates()

    for template in brick_templates:
        brickInstance = BrickInstance(template, get_random_transformation(), 15)
        write_bricks_to_file(
            [brickInstance],
            file_path=debugger.file_path(f"{template.id}_test.ldr"),
            debug=True,
        )

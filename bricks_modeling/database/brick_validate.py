from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.bricks.brickinstance import BrickInstance
import numpy as np
from scipy.spatial.transform import Rotation as R


def gen_random_rotation():
    rand_rot_axis = np.random.rand(3)
    rot = R.from_rotvec(np.random.randint(0, 100) * rand_rot_axis)
    return rot.as_matrix()

if __name__ == "__main__":
    debugger = MyDebugger("test")
    brick_templates, template_ids = get_all_brick_templates()

    for template in brick_templates:
        trans_matrix = np.identity(4, dtype=float)
        trans_matrix[:3,:3] = gen_random_rotation()
        trans_matrix[:3, 3] = 100*np.random.rand(3)
        brickInstance = BrickInstance(template, trans_matrix, 15)
        write_bricks_to_file(
            [brickInstance],
            file_path=debugger.file_path(f"{template.id}_test.ldr"),
            debug=True,
        )

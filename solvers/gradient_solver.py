import json
import numpy as np
from typing import List, Tuple

from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.bricks.brickinstance import BrickInstance

from util.geometry_util import point_local2world, vec_local2world

if __name__ == "__main__":
    print("gradient solver")
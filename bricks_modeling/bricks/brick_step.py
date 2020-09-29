from typing import List
from bricks_modeling.bricks.brickinstance import BrickInstance

class BrickStep:
    def __init__(self, bricks: List[BrickInstance]):
        self.bricks = bricks

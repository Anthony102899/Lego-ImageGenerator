from typing import List

from bricks.brickinstance import BrickInstance


def write_bricks_to_file(bricks: List[BrickInstance], file_path):
    file = open(file_path, "a")
    ldr_content = "\n".join([brick.to_ldraw() for brick in bricks])
    file.write(ldr_content)
    file.close()
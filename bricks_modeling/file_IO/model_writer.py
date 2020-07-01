from typing import List

from bricks_modeling.bricks.brickinstance import BrickInstance


def write_bricks_to_file(bricks: List[BrickInstance], file_path, debug=False):
    file = open(file_path, "a")
    ldr_content = "\n".join([brick.to_ldraw() for brick in bricks])
    if debug:
        conn_point = "\n".join(
            [c.to_ldraw() for brick in bricks for c in brick.get_current_conn_points()]
        )
        ldr_content = ldr_content + "\n" + conn_point
    file.write(ldr_content)
    file.close()
    print(f"file {file_path} saved!")

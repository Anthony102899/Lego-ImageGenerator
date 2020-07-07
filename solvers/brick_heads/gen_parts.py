from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
import os

if __name__ == "__main__":
    debugger = MyDebugger("test")
    template_bricks = read_bricks_from_file("./data/brickheads/components_old/template.ldr", read_fake_bricks = True)

    total_bricks = template_bricks

    test_dir = "./data/brickheads/components_old/"
    for file_name in os.listdir(test_dir):
        addition_bricks = []
        modifier_bricks = read_bricks_from_file(test_dir + file_name, read_fake_bricks=True)
        for b in modifier_bricks:
            if b not in template_bricks:
                addition_bricks.append(b)

        write_bricks_to_file(
            addition_bricks, file_path=debugger.file_path(file_name), debug=False
        )
        total_bricks = total_bricks + addition_bricks

    write_bricks_to_file(
        total_bricks, file_path=debugger.file_path("complete.ldr"), debug=False
    )
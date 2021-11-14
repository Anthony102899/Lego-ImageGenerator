from bricks_modeling.file_IO.model_writer import write_bricks_to_file_with_steps, write_model_to_file
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_reader import read_model_from_file, read_bricks_from_file

'''
We assume the following information is provided:
1) assembly order
2) grouping
3) default camera view
'''

if __name__ == "__main__":
    debugger = MyDebugger("brick_heads")
    file_path = r"data/full_models/steped_talor.ldr"

    model = read_model_from_file(file_path, read_fake_bricks=True)
    write_model_to_file(model, debugger.file_path(f"complete_full.ldr"))
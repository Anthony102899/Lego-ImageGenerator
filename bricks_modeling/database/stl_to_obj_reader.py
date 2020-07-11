import os
from os import path

from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.database.ldr_to_stl_reader import (
    get_file_name_in_a_directory_with_suffix,
)
from bricks_modeling.database.stl2obj import stl2obj


def convert_stl_to_obj(stl_file_name, stl_directory, obj_directory, debug):
    # ldr_file_name must under ldr_directory
    brick_templates, template_ids, _ = get_all_brick_templates()
    if debug:
        if stl_file_name in template_ids:
            if stl_file_name in get_file_name_in_a_directory_with_suffix(
                stl_directory, ".obj"
            ):
                print(f"{stl_file_name} in {obj_directory}")
            else:
                print(f"{stl_file_name} not in {obj_directory}")
        return
    if (
        stl_file_name in template_ids
        and stl_file_name
        not in get_file_name_in_a_directory_with_suffix(stl_directory, ".obj")
    ):
        stl2obj(
            f'{stl_directory + "/" + stl_file_name + ".stl"}',
            f'{obj_directory + "/" + stl_file_name + ".obj"}',
        )
        print(f"new file{stl_file_name}.obj has been created")


def convert_stls_to_objs(stl_directory, obj_directory, debug=False):
    for stl_file_name in get_file_name_in_a_directory_with_suffix(
        stl_directory, ".stl"
    ):
        convert_stl_to_obj(stl_file_name, stl_directory, obj_directory, debug)


if __name__ == "__main__":
    stl_direcotry = os.path.join(path.dirname(__file__), "stl")
    stl_single_parts_direcotry = os.path.join(path.dirname(__file__), "stl", "s")

    obj_directory = os.path.join(path.dirname(__file__), "obj")
    obj_directory_s = os.path.join(path.dirname(__file__), "obj", "s")

    print("Updating obj directory")

    convert_stls_to_objs(stl_direcotry, obj_directory)
    convert_stls_to_objs(stl_single_parts_direcotry, obj_directory_s)

    print("obj directory up to date")

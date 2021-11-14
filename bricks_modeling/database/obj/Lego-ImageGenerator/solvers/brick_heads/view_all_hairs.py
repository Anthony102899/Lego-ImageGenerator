from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from util.debugger import MyDebugger
import os
from solvers.brick_heads.part_selection import get_bricks_from_files
import solvers.brick_heads.config as conf

import solvers.brick_heads.bach_render_images as render


def test_online_models():
    debugger = MyDebugger("test")
    template_bricks = read_bricks_from_file(
        r"/Users/apple/Dropbox/deecamp/parts_library/template.ldr", read_fake_bricks=True
    )

    test_dir = "hair"

    for gender in ["M","F"]:
        for i in range(0,5):
            total_bricks = []
            total_bricks += template_bricks
            for file_name in {f"{test_dir}/{gender}-Hair-{i}", f"{test_dir}/{gender}-Hair-{i}_lh"}:
                bricks = get_bricks_from_files([file_name], 0)
                if os.path.exists(os.path.join(conf.parts_dir, f"{file_name}_skin" + ".ldr")):
                    skin_bricks = get_bricks_from_files([f"{file_name}_skin"], 19)

                total_bricks += bricks
                total_bricks += skin_bricks

            write_bricks_to_file(
                total_bricks,
                file_path=debugger.file_path(f"{gender}-Hair-{i}.ldr"),
                debug=False,
            )

    render.render_ldrs(debugger._debug_dir_name)


def show_testing_models():
    debugger = MyDebugger("test")
    template_bricks = read_bricks_from_file(
        r"/Users/apple/Dropbox/deecamp/parts_library/template.ldr", read_fake_bricks=True
    )

    test_dir = "hair/testing"

    for hair_name in {1, 2, 3, 4}:
        total_bricks = []
        total_bricks += template_bricks
        for file_name in {f"{test_dir}/{hair_name}", f"{test_dir}/{hair_name}_lh"}:
            bricks = get_bricks_from_files([file_name], 0)
            if os.path.exists(os.path.join(conf.parts_dir, f"{file_name}_skin" + ".ldr")):
                skin_bricks = get_bricks_from_files([f"{file_name}_skin"], 19)

            total_bricks += bricks
            total_bricks += skin_bricks

        write_bricks_to_file(
            total_bricks,
            file_path=debugger.file_path(f"{hair_name}.ldr"),
            debug=False,
        )

    render.render_ldrs(debugger._debug_dir_name)

if __name__ == "__main__":
    test_online_models()



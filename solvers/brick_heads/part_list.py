from bricks_modeling.file_IO.model_writer import write_bricks_to_file_with_steps
from util.debugger import MyDebugger
import os
import csv
import solvers.brick_heads.bach_render_images as render
import solvers.brick_heads.part_selection as p_select
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
import numpy as np
import xlsxwriter
from typing import Dict, List
from bricks_modeling.database.ldraw_colors import read_colors

def gen_part_images(temp_dir, ldraw_ids: List[str]):
    image_dir = os.path.join(temp_dir, "images")
    os.mkdir(image_dir)

    for (id, color) in ldraw_ids:
        file = open(os.path.join(image_dir, f"{id}_{color}.ldr"), "a")
        if len(id.split("_")) > 1:  # special brick with texture
            ldraw_id = id.split("_")[0]
        else:
            ldraw_id = id
        ldr_content = f"1 {color} 0 0 0 1 0 0 0 1 0 0 0 1 {ldraw_id}.dat"
        file.write(ldr_content)
        file.close()
        print(f"file {os.path.join(image_dir, f'{id}_{color}.ldr')} saved!")

    render.render_ldrs(image_dir)

    return image_dir


def generate_part_list(file_path, brick_count: Dict, brick_occurrence:Dict, part_images_dir):

    # Create an new Excel file and add a worksheet.
    workbook = xlsxwriter.Workbook(file_path)
    worksheet = workbook.add_worksheet()

    # Widen the first column to make the text clearer.
    worksheet.set_column("D:D", 12)
    worksheet.set_column("F:F", 17)
    for i in range(0, len(brick_count)):
        worksheet.set_row(i + 1, 60)

    worksheet.write(0, 0, "brick ID")
    worksheet.write(0, 1, "color ID")
    worksheet.write(0, 2, "color name")
    worksheet.write(0, 3, "brick count")
    worksheet.write(0, 4, "brick occurrence")
    worksheet.write(0, 5, "brick image")

    rgb2id, id2name = read_colors()
    for idx, (key, value) in enumerate(brick_count.items()):
        ldraw_id, color = key
        worksheet.write(idx + 1, 0, ldraw_id)
        worksheet.write(idx + 1, 1, color)
        worksheet.write(idx + 1, 2, id2name[color])
        worksheet.write(idx + 1, 3, value)
        worksheet.write(idx + 1, 4, brick_occurrence[key])
        worksheet.insert_image(
            idx + 1,
            5,
            os.path.join(part_images_dir, f"{ldraw_id}_{color}.ldr.png"),
            {
                "x_scale": 0.16,
                "y_scale": 0.16,
                "object_position": 1,
                "x_offset": 10.0,
                "y_offset": 1.0,
            },
        )

    workbook.close()


if __name__ == "__main__":
    debugger = MyDebugger("brick_heads")
    dir_path = r"/Users/apple/workspace/lego-photo-studio/debug/2020-08-04_11-35-00_brick_heads"

    final_str = ""
    brick_count = {}

    for i in range(1, 108):
        file_path = os.path.join(dir_path, f"complete_{i}.ldr")
        bricks = read_bricks_from_file(file_path, read_fake_bricks=True)
        for b in bricks:
            if len(b.template.id.split("_")) > 1:  # special brick with texture
                ldraw_id = b.template.id.split("_")[0]
            else:
                ldraw_id = b.template.id

            if (ldraw_id, b.color) not in brick_count:
                brick_count[(ldraw_id, b.color)] = 1
            else:
                brick_count[(ldraw_id, b.color)] += 1

    brick_occurrence = dict.fromkeys(brick_count.keys(), 0)
    for i in range(1, 108):
        file_path = os.path.join(dir_path, f"complete_{i}.ldr")
        bricks = read_bricks_from_file(file_path, read_fake_bricks=True)
        for key in brick_count.keys():
            for b in bricks:
                if len(b.template.id.split("_")) > 1:  # special brick with texture
                    ldraw_id = b.template.id.split("_")[0]
                else:
                    ldraw_id = b.template.id

                if (ldraw_id, b.color) == key:
                    brick_occurrence[key] += 1
                    break

    for key, value in brick_count.items():
        print(f"{key}\t{value}")

    for key, value in brick_occurrence.items():
        print(f"{key}\t{value}")

    image_dir = gen_part_images(debugger._debug_dir_name, list(brick_count.keys()))
    generate_part_list(debugger.file_path("part_list.xlsx"), brick_count, brick_occurrence, image_dir)

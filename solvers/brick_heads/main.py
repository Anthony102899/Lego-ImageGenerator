from bricks_modeling.file_IO.model_writer import write_bricks_to_file_for_instruction
from util.debugger import MyDebugger
import os
import csv
import solvers.brick_heads.bach_render_images as render
import solvers.brick_heads.part_selection as p_select

def string_to_tuple_list(tuple_str):
    if tuple_str == "[]":
        return []
    else:
        str_list = tuple_str.split("), (")
        str_list[0] = str_list[0][2:]
        str_list[-1] = str_list[-1][:-2]
        result_list = []
        for t in str_list:
            numbers = t.split(", ")
            if len(numbers) == 3:
                result_list.append((
                    int(numbers[0]),
                    int(numbers[1]),
                    int(numbers[2])
                ))
            elif len(numbers) == 2:
                result_list.append((
                    int(numbers[0]),
                    int(numbers[1])
                ))

        return result_list

def string_to_str_list(tuple_str):
    if tuple_str == "[]":
        return []
    else:
        str_list = tuple_str.split("', '")
        str_list[0] = str_list[0][2:]
        str_list[-1] = str_list[-1][:-2]

        return str_list


if __name__ == "__main__":
    debugger = MyDebugger("brick_heads")

    csv_path = os.path.join(os.path.dirname(__file__), "input_data", "data.csv")

    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        csv_matrix = list(csv_reader)

    for model in csv_matrix[54:55]:
        if model[1] == "":
            continue
        bricks, texture_brick_strs, ordered_bricks = p_select.gen_LEGO_figure(
            gender=int(model[1]),
            # gender=0,
            # hair=int(model[2]),
            hair=5,
            hair_color=(
                int(model[3].split(",")[0]),
                int(model[3].split(",")[1]),
                int(model[3].split(",")[2]),
            ),
            # bang=int(model[4]),
            bang=0,
            skin_color=int(model[5]),
            eye=int(model[6]),
            jaw=int(model[7]),
            hands=int(model[8]),
            clothes_style=int(model[9]),
            # clothes_bg_color = string_to_tuple_list(model[10]),
            clothes_bg_color=[],
            logo_imgs =string_to_str_list(model[11]),
            logo_pos =string_to_tuple_list(model[12]),
            pants_type = int(model[13]),
            # pants_color = string_to_tuple_list(model[14]),
            pants_color=[],
        )

        ### hack color for LKF
        # for b_list in ordered_bricks:
        #     for b in b_list:
        #         if b.template.id == "22885":
        #             b.color = 7
        #         if b.template.id == "3022":
        #             b.color = 0
        #         if b.template.id == "3941":
        #             b.color = 72

        write_bricks_to_file_for_instruction(
            ordered_bricks,
            file_path=debugger.file_path(f"complete_{int(model[0])}.ldr"),
            texture_brick_strs=texture_brick_strs
        )

    # render.render_ldrs(debugger._debug_dir_name)

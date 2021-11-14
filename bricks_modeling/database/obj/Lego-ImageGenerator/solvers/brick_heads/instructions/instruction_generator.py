from bricks_modeling.bricks.model import Model
from bricks_modeling.file_IO.util import to_ldr_format

def write_model_to_file(model: Model, file_path):
    file = open(file_path, "a")
    ldr_content = ""

    for group_name in model.group_names:
        current_group = model.groups[group_name]
        ldr_content += f"0 FILE {group_name}\n"
        for brick_step in current_group.brick_steps:
            ldr_content += "0 STEP\n"
            # output all bricks
            for b in brick_step.bricks:
                ldr_content += (b.to_ldraw() + "\n")
            # output all subgroups
            for i in range(len(brick_step.subgroup_names)):
                name      = brick_step.subgroup_names[i]
                color     = brick_step.subgroups_colors[i]
                trans_mat = brick_step.subgroups_transformation[i]
                ldr_content += (to_ldr_format(color, trans_mat, name) + "\n")
        ldr_content += f"0 NOFILE\n"

    file.write(ldr_content)
    file.close()
    print(f"file {file_path} saved!")

from bricks_modeling.database import ldraw_colors

def select_nearest_face_color(rgb):
    skin_color_map = {
        (244,244,244):511, # white
        (255,201,149):78, # yellow
        (145,80,28):484  # black or 10484?
    }
    return select_nearest_color(rgb, given_list=skin_color_map)

def select_nearest_color(rgb, given_list = None):
    if given_list is None:
        all_colors = ldraw_colors.read_colors()
    else:
        all_colors = given_list

    best_id = -1
    closest_dist = 1e8
    for l_rgb, color_id in all_colors.items():
        current_dist = (rgb[0]-l_rgb[0])**2+(rgb[1]-l_rgb[1])**2+(rgb[2]-l_rgb[2])**2
        if current_dist < closest_dist:
            best_id = color_id
            closest_dist = current_dist

    return best_id

def get_part_files(body, json_data):
    if body == "hair":
        gender = "F" if json_data["gender"] == 0 else "M"
        length = json_data["hair"][1]
        nearest_ldr_color = select_nearest_color(rgb=json_data["hair"][0])

        if sum(json_data["hair"][2])==0: # no bang
            return [(f"{gender}-Hair-{length}", nearest_ldr_color),
                    ("Hair_no_lh", nearest_ldr_color)]
        else:
            return [(f"{gender}-Hair-{length}",nearest_ldr_color),
                    (f"{gender}-Hair-{length}_lh", nearest_ldr_color)]

    elif body == "right_arm":
        return [("right_arm_0", None)]

    elif body == "left_arm":
        return [("left_arm_0", None)]

    elif body == "clothes":
        nearest_ldr_color = select_nearest_color(rgb=json_data["clothes"][0])
        return [("clothes", nearest_ldr_color)]

    elif body == "glasses":
        if json_data["glasses"] == -1:
            return [("eyes_0", None)]
        else:
            return [("eyes_glasses", None)]

    elif body == "beard":
        return [("mustache_no" if json_data["beard"] == -1 else "mustache_yes", None)]

    else:
        print("error id:", body_id)
        input()
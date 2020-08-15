import os

def read_colors():
    results = dict()
    color_file = "./bricks_modeling/database/my_LDConfig.ldr"
    f = open(color_file, "r")

    for line in f.readlines():
        if line.startswith("0 !COLOUR"):
            line_content = line.rstrip().split()
            if len(line_content) <= 9 or (len(line_content) > 9 and line_content[9] != "ALPHA"): # ignore transparent color
                color = (int(line_content[6][1+0*2:1+0*2+2], 16),
                         int(line_content[6][1+1*2:1+1*2+2], 16),
                         int(line_content[6][1+2*2:1+2*2+2], 16))
                results[color] = int(line_content[4])

    return results

def color_phraser(
    file_path=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "database", "ldraw", "LDConfig.ldr"
    )
):
    color_dict = {}

    f = open(file_path, "r")
    for line in f.readlines():
        line_content = line.rstrip().split()
        if len(line_content) < 8 or line_content[1] == "//":
            continue
        else:
            color_dict[line_content[4]] = [
                int(line_content[6][i : i + 2], 16) / 255 for i in (1, 3, 5)
            ]
            #print(f"color {line_content[4]} is {color_dict[line_content[4]]}")

    return color_dict

if __name__ == "__main__":
    read_colors()
import base64
from string import Template
from util.debugger import MyDebugger
import os

test_template = Template(
    """sss$which
sdsd"""
)
out_template = Template(
 """0 FILE $name.dat
0 $name
0 Name: $name.dat
0 !LICENSE Redistributable under CCAL version 2.0 : see CAreadme.txt
0 BFC CERTIFY
0 PE_TEX_PATH -1
0 PE_TEX_INFO $image
3 16 $x_low $y_low $z_shift $x_high $y_low $z_shift $x_high $y_high $z_shift 1 1 0 1 0 0
3 16 $x_low $y_low $z_shift $x_high $y_high $z_shift $x_low $y_high $z_shift 1 1 0 0 1 0
1 16 0 0 0 1 0 0 0 1 0 0 0 1 $part.dat
"""
)

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def image_to_lego_texture(image_path, part_id, image_name, x_low, y_low, x_high, y_high, z_shift):
    return out_template.substitute(
        name=image_name,
        image=image_to_base64(image_path),
        x_low=x_low,
        y_low = y_low,
        x_high = x_high,
        y_high = y_high,
        z_shift = z_shift,
        part=part_id,
    )

if __name__ == "__main__":
    debugger = MyDebugger("brick_heads")

    # image_to_lego_texture(r"/Users/apple/workspace/lego-photo-studio/solvers/brick_heads/parts/textures/1.png",
    #                       "3245c", "004", -20, 0, 20, 48, 10.1, debugger.file_path(""))
    image_to_lego_texture(r"/Users/apple/workspace/lego-photo-studio/solvers/brick_heads/parts/textures/14.png",
                          "3010", "001", -40, 0, 40, 24, 10.2, debugger.file_path(""))


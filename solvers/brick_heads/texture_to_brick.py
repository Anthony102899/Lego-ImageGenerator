import base64
from string import Template
test_template = Template("""sss$which
sdsd""")
out_template = Template("""0 FILE $name.dat
0 $name
0 Name: $name.dat
0 !LICENSE Redistributable under CCAL version 2.0 : see CAreadme.txt
0 BFC CERTIFY
0 PE_TEX_PATH -1
0 PE_TEX_INFO $image
3 16 $scale -0.01 -$scale $scale -0.01 $scale -$scale -0.01 $scale 0 0 1 0 1 1
3 16 $scale -0.01 -$scale -$scale -0.01 $scale -$scale -0.01 -$scale 0 0 1 1 0 1
1 16 0 0 0 1 0 0 0 1 0 0 0 1 $part.dat
""")


def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def image_to_lego_texture(image_path, part_id, image_name, image_scale, out_path):
    with open(out_path+image_name+".dat", "w") as out_file:
        out_file.write(out_template.substitute(name=image_name, image=image_to_base64(
            image_path), scale=image_scale, part=part_id))
        out_file.close()


# image_to_lego_texture("F:\Research\Lego studio\icon.png",
#                       "3068b", "icon", 20, "./")

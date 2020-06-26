import os
from os import path

from bricks_modeling.bricks.brick_factory import get_all_brick_templates

'''
    1.You need perl environment to run main, install perl for your device, and then install the perl plugin for pycharm and REBOOT(important).
    
    2.When you add a new brick in the database, you can run main to update the stl database.
    
    3.This program is slow (around 1 minute), need further modification if needed.
    
    4. Note: in parts file, there is also a "s" directory containing ldrs, so the stl directory follows this convention.
      
'''
def get_file_name_in_a_directory(file_dir) -> []:
    file_Names=[]
    for files in os.listdir(file_dir):
        if os.path.splitext(files)[1] == '.dat' or os.path.splitext(files)[1] == '.stl': #Get file name end wtih dat or stl
            file_Names.append(os.path.splitext(files)[0])
    return file_Names

def convert_ldr_to_stl(ldr_file_name, ldr_directory, stl_directory, debug):
    #ldr_file_name must under ldr_directory
    brick_templates, template_ids = get_all_brick_templates()
    if (debug):
        if ldr_file_name in template_ids:
            if ldr_file_name in get_file_name_in_a_directory(stl_directory):
                print(f"{ldr_file_name} in {stl_directory}")
            else:
                print(f"{ldr_file_name} not in {stl_directory}")
        return
    if ldr_file_name in template_ids and ldr_file_name not in get_file_name_in_a_directory(stl_directory):
        os.system(f'perl ldraw2stl/bin/dat2stl --file {ldr_directory +"/"+ ldr_file_name + ".dat"} --ldrawdir ./ldraw > {stl_directory + "/" + ldr_file_name}.stl')
        print(f"new file{ldr_file_name}.stl has been created")


def convert_ldrs_to_stls(ldr_directory, stl_directory, debug=False):
    for ldr_file_name in get_file_name_in_a_directory(ldr_directory):
        convert_ldr_to_stl(ldr_file_name,ldr_directory, stl_directory, debug)

if __name__ == "__main__":

    LEGO_parts_direcotry = os.path.join(path.dirname(__file__) ,"ldraw" ,"parts")
    LEGO_single_parts_direcotry = os.path.join(path.dirname(__file__) ,"ldraw" ,"parts","s")

    stl_directory = os.path.join(path.dirname(__file__) ,"stl")
    stl_directory_s = os.path.join(path.dirname(__file__), "stl","s")

    print("Updating stl directory, this may last for a minute")

    convert_ldrs_to_stls(LEGO_parts_direcotry, stl_directory)
    convert_ldrs_to_stls(LEGO_single_parts_direcotry, stl_directory_s)

    print("stl directory up to date")

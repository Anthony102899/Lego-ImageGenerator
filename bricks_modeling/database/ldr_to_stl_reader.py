import os
from os import path

from bricks_modeling.bricks.brick_factory import get_all_brick_templates

'''
    You need perl environment to run main
'''
def get_file_name_in_a_directory(file_dir) -> []:
    file_Names=[]
    for files in os.listdir(file_dir):
        #print(files)
        if os.path.splitext(files)[1] == '.dat':
            file_Names.append(os.path.splitext(files)[0])
    return file_Names

def convert_ldr_to_stl(ldr_file_name, ldr_directory, stl_directory):
    #ldr_file_name must under ldr_directory
    brick_templates, template_ids = get_all_brick_templates()
    if ldr_file_name in template_ids:
        os.system(f'perl ldraw2stl/bin/dat2stl --file {ldr_directory +"/"+ ldr_file_name + ".dat"} --ldrawdir ./ldraw > {stl_directory + "/" + ldr_file_name}.stl')
        print(f"file{ldr_file_name} pass")


def convert_ldrs_to_stls(ldr_file_names:[], ldr_directory, stl_directory):
    for ldr_file_name in ldr_file_names:
        convert_ldr_to_stl(ldr_file_name,ldr_directory,stl_directory)

if __name__ == "__main__":
    '''
        Note: in parts file, there is also a "s" directory containing ldrs, so the stl directory follows this convention
    '''
    LEGO_parts_direcotry = os.path.join(path.dirname(__file__) ,"ldraw" ,"parts")
    LEGO_single_parts_direcotry = os.path.join(path.dirname(__file__) ,"ldraw" ,"parts","s")

    stl_directory = os.path.join(path.dirname(__file__) ,"stl")
    stl_directory_s = os.path.join(path.dirname(__file__), "stl","s")

    file_names_in_parts = get_file_name_in_a_directory(LEGO_parts_direcotry)#file name in lego-solver/bricks_modeling/database\ldraw\parts
    file_names_in_parts_s = get_file_name_in_a_directory(LEGO_single_parts_direcotry)#lego-solver/bricks_modeling/database\ldraw\parts\s

    convert_ldrs_to_stls(file_names_in_parts, LEGO_parts_direcotry, stl_directory)
    convert_ldrs_to_stls(file_names_in_parts_s,LEGO_single_parts_direcotry,stl_directory_s)

import sys
import string
import os.path
import getopt
from os import path

'''

    I do not understand this code, but it works.

    Source code is python2, I translate parts of them(uncommented part) to python3.

    while inside while, very similar to what I did before......
    
'''

#!/usr/bin/env python
#
# stl2obj.py: Script to convert .stl ASCII (3D systems) mesh into .obj ASCII mesh (wavefront).
#
#
# Copyright (C) 2010  Clement Creusot
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



'''def print_help():
    print
    "Usage: " + os.path.basename(sys.argv[0]) + " [OPTIONS] filein.stl"
    print
    "   Options: -o OUTPUT_FILE"
    print
    "               Write the output mesh in OUTPUT_FILE"
    print
    "            -b Use binary STL representation"
    print
    "            -f Fast (dont try to find multiply defined point"
    print
    "               , create 3 points per facet)"
    sys.exit()


def print_error(*str):
    print
    "ERROR: ",
    for i in str:
        print
        i,
    print
    sys.exit()


# Reading the parameters
#try:
optlist, args = getopt.getopt(sys.argv[1:], 'hbfo:')
#except getopt.GetoptError, err:
    #print
    #str(err)
    #print_help()

if len(args) < 1:
    print_help()

# verify the argument is an stl file
stlfilename = args[0]
if string.split(stlfilename, ".")[-1].lower() != "stl":
    print_error(stlfilename, ": The file is not an .stl file.")
if not os.path.exists(stlfilename):
    print_error(stlfilename, ": The file doesn't exist.")

# By default the output is the stl filename followed by '.obj'
objfilename = stlfilename + ".obj"

# use options
useBinary = False
useFast = False
for opt, value in optlist:
    if opt in ("-h", "--help"):
        print_help()
    elif opt == '-o':
        objfilename = value
    elif opt == '-b':
        useBinary = True
        print_error("Binary read not yet implemented.")
    elif opt == '-f':
        useFast = True
    else:
        print("Unhandled option")
        print_help()
'''
useFast = True

def stl2obj(stlfilename, objfilename):

    pointList = []
    facetList = []


    def GetPointId(point, list):
        global useFast
        if not useFast:
            for i, pts in enumerate(list):
                if pts[0] == point[0] and pts[1] == point[1] and pts[2] == point[2]:
                    # obj start to count at 1
                    return i + 1
        list.append(point)
        # obj start to count at 1
        return len(list)


    # start reading the STL file
    stlfile = open(stlfilename, "r")
    line = stlfile.readline()
    lineNb = 1
    while line != "":
        tab = line.rstrip().split()
        if len(tab) > 0:
            if tab[0] == "facet":
                print
                lineNb
                vertices = []
                normal = map(float, tab[2:])
                while tab[0] != "endfacet":
                    if tab[0] == "vertex":
                        pts = map(float, tab[1:])
                        vertices.append(GetPointId(pts, pointList))
                    line = stlfile.readline()
                    lineNb = lineNb + 1
                    tab = line.rstrip().split()
                if len(vertices) == 0:
                    print("Unvalid facet description at line ")
                facetList.append({"vertices": vertices, "normal": normal})

        line = stlfile.readline()
        lineNb = lineNb + 1

    stlfile.close()

    # Write the target file
    objfile = open(objfilename, "w")
    objfile.write("# File type: ASCII OBJ\n")
    objfile.write("# Generated from " + os.path.basename(stlfilename) + "\n")

    for pts in pointList:
        objfile.write("v " + " ".join(map(str, pts)) + "\n")

    for f in facetList:
        objfile.write("f " + " ".join(map(str, f["vertices"])) + "\n")

    objfile.close()

#if __name__ == "__main__":
    #stl_file = os.path.join(path.dirname(__file__),"2431.stl")
    #obj_file = os.path.join(path.dirname(__file__),"2431.obj")
    #stl2obj(stl_file, obj_file)
import os
import sys
import argparse
import numpy as np
import common.faces as faces
import common.dataset as ds
from PIL import Image, ImageDraw


def get_rectangle( points ):
    print(points)
    maxs = np.amax(points, axis=0)
    mins = np.amin(points, axis=0)
    print(maxs)
    print(mins)
    dx = (maxs[0] - mins[0])
    dy = ( maxs[1] - mins[1] )
    rectangle = [mins[0] - dx / 2, maxs[1] - dy * 2, mins[0]+ (dx * 1.5), maxs[1] + (dy * 2)]
    # rectangle.append( (mins[0], maxs[1] ) )
    ## rectangle.append( maxs[0] - mins[0])
    ## rectangle.append( maxs[1] - mins[1])
    # rectangle.append( (maxs[0], mins[1] ) )



    return rectangle


parser = argparse.ArgumentParser(description="Prepare image dataset resinzing and organizing by label")
parser.add_argument('dirs', metavar='dirs', type=str, nargs='+', help='list of directories with images')
parser.add_argument('-o', '--output', default='output',
                    help='Output directory where the images will be classified. Default to output in current dir')

args = parser.parse_args()
print("Args", args)
dirs = args.dirs
output = args.output
for dir in dirs:
    facesLib = faces.Faces(dir)
    faces = facesLib.getImages()['objects']
    for face in faces:
        pillowImage = Image.fromarray(face["image"])
        rectangle = get_rectangle( face["landmarks"]["bottom_lip"] + face["landmarks"]["top_lip"])
        pillowImage2 = pillowImage.crop( rectangle )
        pillowImage2.save(output +  "/" + face["name"] + ".jpg")


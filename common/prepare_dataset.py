from __future__ import print_function
import os
import sys
import argparse
real_path = os.path.realpath(__file__)
base_dir = real_path[:real_path.rfind("/")]
import dataset as ds
from imgaug import augmenters as iaa

def process_augmentations( augmentationString ):
    processedAugmentations = []
    if augmentationString:
        augmentations = augmentationString.split(',')
        for augmentation in augmentations:
            method, values = augmentation.split(":")

            func = getattr(iaa, method)

            values = map(float, values.split(';'))

            if len(values) == 1:
                augmenter = func(values[0])
            else:
                augmenter = func(*values)

            print("Got augmenter:")
            print(augmenter)
            processedAugmentations.append(augmenter)

    return processedAugmentations

parser = argparse.ArgumentParser(description="Prepare image dataset resinzing and organizing by label")
parser.add_argument('dirs', metavar='dirs', type=str, nargs='+', help='list of directories with images')
parser.add_argument('labels', metavar='l', type=str, help='comma separated labels for each directory associated')
parser.add_argument('-o', '--output', default='output',
                    help='Output directory where the images will be classified. Default to output in current dir')
parser.add_argument('-b', '--boxes',type=bool, default=False,
                    help='This option avoids resizing to use the prepare for object detection')
parser.add_argument('--augmentations', '-a', metavar="a", type=str, help="Comma separated augmentations for the "
                                                                         "dataset, like GaussianBlur:0.6,Fliplr:0.5,Sharpen:0.9;0.3 "
                                                                         "... "
                                                                         "To see the list of parameters, check "
                                                                         "https://github.com/aleju/imgaug")

# python prepare_dataset.py /Users/alvaroescarcha/Desktop/tshirt/tshirts 0 -o /Users/alvaroescarcha/Desktop/tshirt_aug10 --augmentations "Sharpen:0.9;0.3,GaussianBlur:1.6,Fliplr:0.5"
# For bounding boxes
#  python prepare_dataset.py /Users/alvaroescarcha/Desktop/tshirt/tshirts 0 -o /Users/alvaroescarcha/Desktop/tshirt_aug10 -b true --augmentations "GaussianBlur:1.0"

args = parser.parse_args()
print("Args", args)
dirs = args.dirs
labels = args.labels.split(',')
labels_dict = {}
for name, label in zip(dirs, labels):
        labels_dict[name]=int(label)

# dirs = [
#         base_dir + "/../dragontrainer/trainImages/geo",
#         base_dir + "/../dragontrainer/trainImages/smoke",
#         base_dir + "/../dragontrainer/trainImages/people",
# ]
# labels_dict = {"smoke": 1, "people": 0, "geo": 0}
print("Label dict", labels_dict)
ds.organize_dirs_with_labels(dirs, labels_dict, args.output, process_augmentations(args.augmentations), args.boxes)




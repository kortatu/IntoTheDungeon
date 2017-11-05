import os
import sys
import argparse
real_path = os.path.realpath(__file__)
base_dir = real_path[:real_path.rfind("/")]
import dataset as ds
from imgaug import augmenters as iaa


def process_augmentations(augmentation_string):
    processed_augmentations = []
    if augmentation_string:
        augmentations = augmentation_string.split(',')
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
            processed_augmentations.append(augmenter)

    return processed_augmentations


parser = argparse.ArgumentParser(description="Prepare image dataset resinzing and organizing by label")
parser.add_argument('dirs', metavar='dirs', type=str, nargs='+', help='list of directories with images')
parser.add_argument('labels', metavar='l', type=str, help='comma separated labels for each directory associated')
parser.add_argument('-o', '--output', default='output',
                    help='Output directory where the images will be classified. Default to output in current dir')
parser.add_argument('-s', '--shape', default=416, # (416 = 12 * 2^5)
                    help='Shape of the target image (just a value that will be used as height and width)')
parser.add_argument('--augmentations', '-a', metavar="a", type=str, help="Comma separated augmentations for the "
                                                                         "dataset, like GaussianBlur:0.6,Fliplr:0.5,Sharpen:0.9;0.3 "
                                                                         "... "
                                                                         "To see the list of parameters, check "
                                                                         "https://github.com/aleju/imgaug")
parser.add_argument('-r', '--ratio', default=0.8, # (416 = 12 * 2^5)
                    help='Ratio of images in training folder. 1.0 will leave test folder empty')


# python prepare_dataset.py /Users/alvaroescarcha/Desktop/tshirt/tshirts 0 -o /Users/alvaroescarcha/Desktop/tshirt_aug10 --augmentations "Sharpen:0.9;0.3,GaussianBlur:1.6,Fliplr:0.5"

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
# target_shape = (336, 336) #  336 = 2^4 * 3 * 7
# target_shape = (224, 24) #  224 = 2^5 * 7
target_shape_as_int = int(args.shape)
target_shape = (target_shape_as_int, target_shape_as_int)
print("Target shape will be", target_shape)
training_ratio = float(args.ratio)
print("Ratio of images in training", training_ratio)
ds.organize_dirs_with_labels(dirs, labels_dict, args.output,
                             target_shape, process_augmentations(args.augmentations), training_ratio)



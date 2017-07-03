import os
import sys
import argparse
real_path = os.path.realpath(__file__)
base_dir = real_path[:real_path.rfind("/")]
import dataset as ds

parser = argparse.ArgumentParser(description="Prepare image dataset resinzing and organizing by label")
parser.add_argument('dirs', metavar='dirs', type=str, nargs='+', help='list of directories with images')
parser.add_argument('labels', metavar='l', type=str, help='comma separated labels for each directory associated')
parser.add_argument('-o', '--output', default='output',
                    help='Output directory where the images will be classified. Default to output in current dir')

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
ds.organize_dirs_with_labels(dirs, labels_dict, args.output)



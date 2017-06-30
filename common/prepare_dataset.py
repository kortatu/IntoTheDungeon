import os
import sys
real_path = os.path.realpath(__file__)
base_dir = real_path[:real_path.rfind("/")]
import dataset as ds

dirs = [
        base_dir + "/../dragontrainer/trainImages/geo",
        base_dir + "/../dragontrainer/trainImages/smoke",
        base_dir + "/../dragontrainer/trainImages/people",
]
labels_dict = {"smoke": 1, "people": 0, "geo": 0}
ds.load_dirs_with_labels(dirs, labels_dict)
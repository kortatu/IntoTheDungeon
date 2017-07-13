import numpy
import scipy.io
from scipy import misc
import numpy as np
import os
from tensorflow.python.framework import dtypes

categories = {'ski':0, 'epic':1, 'musical':2, 'extreme':3, 'pool':4, 'trump':5, 'nosignal':6, 'advertising':7,"sync":8, "balloons": 9}


class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=True):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0., 1.]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self.one_hot = one_hot
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
        #if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
        #    images = images.astype(numpy.float32)
        #    images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * (len(categories) -1 )
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
            ]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


class PathDataSet(object):

    def __init__(self,
                 paths,
                 labels,
                 train_percentage,
                 dtype=dtypes.float32):
        """Construct a DataSet.
        `dtype` can be either `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0., 1.]`.
        """
        self._dtype = dtypes.as_dtype(dtype).base_dtype
        if self._dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        assert len(paths) == labels.shape[0], (
             'images.shape: %s labels.shape: %s' % (len(paths), labels.shape))

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)

        trainXs, trainYs, testXs, testYs = shuffle_and_slice(paths, labels, train_percentage)
        self._train_paths = trainXs
        self._num_examples = len(trainXs)
        self._train_labels = trainYs
        self._test_paths = testXs
        self._test_labels = testYs
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._test_images = None

    @property
    def train_paths(self):
        return self._train_paths

    @property
    def train_labels(self):
        return self._train_labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._train_paths = self._train_paths[perm]
            self._train_labels = self._train_labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.load_images(self._train_paths[start:end]), self._train_labels[start:end]

    def load_images(self, paths):
        images = []
        for path in paths:
            images.append(misc.imread(path))
        images = np.asarray(images, dtype=np.float32)
        # print("Image paths shape", images.shape)
        images = np.reshape(images, [images.shape[0], -1])
        if self._dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        # print("Images shape flattened", images.shape)
        return images

    def number_of_batches(self, batch_size):
        return len(self._train_paths) / batch_size

    def test_images_and_labels(self, max=None):
        if self._test_images is None:
            print('Loading %04d test images' % len(self._test_paths))
            self._test_images = self.load_images(self._test_paths)
        tests_len = len(self._test_images)
        if max is None:
            max = tests_len
        if max < tests_len:
            perm = numpy.arange(tests_len)
            numpy.random.shuffle(perm)
            self._test_images = self._test_images[perm]
            self._test_labels = self._test_labels[perm]
            self._test_paths  = self._test_paths[perm]
        return self._test_images[:max], self._test_labels[:max], self._test_paths[:max]


def organize_dirs_with_labels(dir_names, labels_dict, output, augmentations = None):
    """Load images and labels of all directories in dir_name. The label
    of each subdirectory will be queried in labels_dict.
    :param dir_names: directory names to scan
    :param labels_dict: dictionary with directory_names to label (could contain several names to the same label)
    :param output directory where the images will be classified. There subdirectories will be created by each label
    :param augmentations Optional parameter to augment the images
    :return: a list of images and a corespondent list of labels for each image
    """
    images = []
    labels = []
    for dirname in dir_names:
        dirname_key = dirname
        print("Dirname key", dirname)
        label = labels_dict[dirname_key]
        print("Label for dirname", dirname_key, "is", label)
        file_names = os.listdir(dirname)
        for file_name in file_names:
            print("File", file_name, "in dir", dirname)
            train_image = load_train_image(dirname, file_name, images, label, labels, True)
            if augmentations:
                train_image = augment_image( train_image, augmentations )
                for augmentation in augmentations:
                    file_name = file_name.replace('.', augmentation.name + '.')
            if train_image is not None:
                save_to_label_folder(output, file_name, label, train_image)

    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels)
    print("Images shape", images.shape)


def load_dirs_with_labels(dir_name):
    paths = []
    labels = []
    categories_in_this_dataset = []
    for dirname, dirnames, filenames in os.walk(dir_name):
        print("Dir name", dirname)
        if dirname != dir_name:
            label = dirname[dirname.rfind("/") + 1:]
            label_int = int(label)
            categories_in_this_dataset.append(label_int)
            print("Label", label)
            for filename in filenames:
                # Just add the path
                full_file_name = os.path.join(dirname, filename)
                # We don't read the image in this moment anymore
                # train_image = misc.imread(full_file_name)
                paths.append(full_file_name)
                labels.append(int(label))

    paths = np.asarray(paths, dtype=str)
    labels = np.asarray(labels)
    labels = np.reshape(labels, labels.shape[0])
    labels = np.eye(len(categories_in_this_dataset))[labels]
    print("Labels shape", labels.shape)
    return paths, labels

def load_dirs(dir_name):
    """Load images and labels of all subdirectories of dir_name. The label
    of each image will be the name of the subdirectory.
    :param dir_name: directory name to scan for sub folders
    :return: a list of images and a corespondent list of labels for each image
    """
    images = []
    labels = []
    for dirname, dirnames, filenames in os.walk(dir_name):
        print("Dir name", dirname)
        if dirname != dir_name:
            label = dirname[dirname.rfind("/") + 1:]
            print("Label", label)
            for filename in filenames:
                load_train_image_old(dirname, filename, images, label, labels)
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels)
    print("Images shape", images.shape)
    images = np.reshape(images, [images.shape[0],-1])
    images = images / 255
    print("Images shape", images.shape)
    print(images[0])
    labels = np.reshape(labels, labels.shape[0])
    labels = np.eye(len(categories))[labels]
    print("Labels shape", labels.shape)
    return images, labels


def load_train_image(dirname, filename, images, label, labels, reshape=False):
    if filename.endswith("jpeg") or filename.endswith("jpg"):
        full_file_name = os.path.join(dirname, filename)
        try:
            train_image = misc.imread(full_file_name)
        except IOError:
            print("Error reading image", filename)
            return
        shape = np.asarray(train_image).shape
        if len(shape) >= 1:
            shape, train_image = normalize_dimensions(shape, train_image)
            target_shape = (300, 400)
            correct_shape = shape[0] == target_shape[0] and shape[1] == target_shape[1]
            correct_shape, train_image = reshape_if_needed(correct_shape, reshape, shape, target_shape, train_image)
            if correct_shape:
                images.append(train_image)
                labels.append(label)
                return train_image
            else:
                print("Incorrect shape", shape)


def save_to_label_folder(output_folder, filename, label, train_image):
    label_dir_name = output_folder + "/" + str(label)
    if not os.path.exists(label_dir_name):
        os.makedirs(label_dir_name)
    misc.imsave(label_dir_name + "/" + filename, train_image)


def reshape_if_needed(correct_shape, reshape, shape, target_shape, train_image):
    if not correct_shape and reshape:
        train_image = reshape_image_adding_bars(train_image, shape, target_shape)
        correct_shape = True
    return correct_shape, train_image


def normalize_dimensions(shape, train_image):
    if len(shape) == 2:
        train_image = to_rgb(train_image)
        shape = np.asarray(train_image).shape
        print("After to_rgb shape", shape)
    return shape, train_image


def load_train_image_old(dirname, filename, images, label, labels):
    if filename.endswith("jpeg") or filename.endswith("jpg"):
        full_file_name = os.path.join(dirname, filename)
        train_image = misc.imread(full_file_name)
        shape = np.asarray(train_image).shape
        if shape[0] == 180 and shape[1] == 240:
            images.append(train_image)
            cat = convert(label)
            labels.append(cat)
        else:
            print("Incorrect shape", shape)


def load_image(filename):
    raw = misc.imread(filename)
    shape = np.asarray(raw).shape
    print("Image shape", shape)
    if shape[0] != 180 or shape[1] != 240:
        raw = reshape_image(raw, shape, filename)
    image = np.asarray(raw, dtype=np.float32) / 255
    image = np.reshape(image, -1)
    print("Unrolled shape", image.shape)
    return image


def reshape_image(raw, shape, filename):
    """ Reshape an image to be evaluated that doesn't fit 180,240. We can resize and
        crop the image if the relation doesn't match
    :param raw: matrix representing the image
    :param shape: shape of the image
    :param filename: reshaped image should be saved with a filename similar to the original one
    :return: matrix representing the image reshaped
    """
    ratio = 180.0 / 240.0
    ratio_2 = float(shape[0]) / float(shape[1])
    print("Original Ratio: ", ratio_2)
    if ratio_2 < ratio:
        print("Cropping rows")
        # Crop y axis diff should be positive
        diff = int(round(shape[1] - (shape[0] / ratio)))
        raw = raw[:, diff/2 :-diff/2, :]
    elif ratio_2 > ratio:
        print("Cropping columns")
        # Crop x axis Diff should be positive
        diff = int(round(shape[0] - (shape[1] * ratio)))
        raw = raw[diff/2 + (diff % 2):-diff/2, :, :]
    shape = np.asarray(raw).shape
    ratio_upd = shape[0] / float(shape[1])
    print("Shape/Ratio after crop: ", shape , ratio_upd)
    im_resize = misc.imresize(raw, (180, 240))
    shape = np.asarray(im_resize).shape
    ratio_upd = shape[0] / float(shape[1])
    print("Shape/Ratio after resize: ", shape , ratio_upd)
    misc.imsave("resize/" + filename+"_res.jpg", im_resize)
    return im_resize
    # print("Image shape is wrong", shape)
    # return None


def reshape_image_adding_bars(raw, shape, target_shape=(180, 240)):
    """ Reshape an image to be evaluated that doesn't fit 180,240. We can resize and
        add black bars if the relation doesn't match
    :param raw: matrix representing the image
    :param shape: shape of the image
    :param target_shape: target shape of the image
    :return: matrix representing the image reshaped
    """
    print("Resizing from " , shape, "to", target_shape)
    ratio_target = float(target_shape[0]) / float(target_shape[1])
    ratio_orig = float(shape[0]) / float(shape[1])
    y = shape[0]
    x = shape[1]
    print("Original Ratio: ", ratio_orig)
    if ratio_orig < ratio_target:
        print("Adding rows")
        # Crop y axis diff should be positive
        y_target = x * ratio_target
        diff = int(round(y_target - y))
        bar_top = diff/2 + diff % 2
        bar_bottom = diff/2
        print("Original shape", raw.shape)
        raw = np.concatenate((np.zeros((bar_top, x, 3)), raw, np.zeros((bar_bottom, x, 3))), 0)
        print("Final shape after adding bars", raw.shape)
    elif ratio_orig > ratio_target:
        print("Adding columns")
        x_target = y / ratio_target
        diff = int(round(x_target - x))
        bar_left = diff/2 + diff % 2
        bar_right = diff/2
        print("Original shape", raw.shape)
        raw = np.concatenate((np.zeros((y, bar_left, 3)), raw, np.zeros((y, bar_right, 3))), 1)
        print("Final shape", raw.shape)

    shape = np.asarray(raw).shape
    ratio_upd = shape[0] / float(shape[1])
    # print("Shape/Ratio after adding bars: ", shape, ratio_upd)
    im_resize = misc.imresize(raw, target_shape)
    shape = np.asarray(im_resize).shape
    ratio_upd = shape[0] / float(shape[1])
    print("Shape/Ratio after resize: ", shape , ratio_upd)
    return im_resize


def to_rgb(img):
    print("***** TO RGB")
    return np.dstack([img.astype(np.uint8),img.astype(np.uint8),img.astype(np.uint8)]).copy(order='C')


def shuffle_and_slice(images, labels, train_percentage=0.8, max_results=None):
    total_samples = images.shape[0]
    perm = numpy.arange(total_samples)
    numpy.random.shuffle(perm)
    shuffled_xs = images[perm]
    shuffled_ys = labels[perm]
    if max_results is not None and max_results < total_samples:
        print("Number of samples to use", max_results)
        total_samples = max_results
    num_training_samples = int(total_samples * train_percentage)
    num_test_samples = total_samples - num_training_samples
    print("Number of training", num_training_samples)
    train_xs = shuffled_xs[:num_training_samples]
    train_ys = shuffled_ys[:num_training_samples]
    print("Number of test", num_test_samples)
    test_xs = shuffled_xs[num_training_samples:total_samples]
    test_ys = shuffled_ys[num_training_samples:total_samples]
    return train_xs, train_ys, test_xs, test_ys


def convert(label):

    return categories[label]


def create_data_sets(file_name):
    """Create the data set for Into The Dungeon"""
    mat = scipy.io.loadmat(file_name)
    print("Keys ", mat.keys())
    all_x = mat['ALLX']
    all_y = mat['ALLY']
    print("Shape of ALLY", all_y.shape)
    all_y = np.reshape(all_y, all_y.shape[0])
    all_y_0_based = all_y - 1
    y_labs = np.eye(len(categories))[all_y_0_based]  # Convert a list of num 0..6 to a list of hot position array [0 0 0 1 0 0 0]
    total_samples = all_y.shape[0]

    print("First value NOT shuffled", y_labs[0])
    perm = numpy.arange(total_samples)
    numpy.random.shuffle(perm)
    shuffled_xs = all_x[perm]
    shuffled_ys = y_labs[perm]
    print("First value shuffled", shuffled_ys[0])

    num_training_samples = int(total_samples * 0.8)
    print("Number of training", num_training_samples)
    train_xs = shuffled_xs[:num_training_samples]
    train_ys = shuffled_ys[:num_training_samples]
    print("Number of test", total_samples - num_training_samples)
    test_xs = shuffled_xs[num_training_samples:]
    test_ys = shuffled_ys[num_training_samples:]
    print("Shape of test", test_xs.shape)
    return train_xs, train_ys, test_xs, test_ys


def augment_image( image, augmentations ):
    if image is not None:
        for augmenter in augmentations:
            image = augmenter.augment_image( image )

    return image
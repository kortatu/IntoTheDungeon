import numpy
import scipy.io
from scipy import misc
import numpy as np
import os
from tensorflow.python.framework import dtypes

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
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

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
                fake_label = [1] + [0] * 9
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

def load_dirs(dir_name):
    images = []
    labels = []
    for dirname, dirnames, filenames in os.walk(dir_name):
        print("Dirname", dirname)
        if "/" in dirname:
            label = dirname[dirname.index("/") + 1:]
            print("Label", label)
            for filename in filenames:
                if filename.endswith("jpeg"):
                    full_file_name = os.path.join(dirname, filename)
                    train_image = misc.imread(full_file_name)
                    shape = np.asarray(train_image).shape
                    if shape[0] == 180 and shape[1] == 240:
                        images.append(train_image)
                        cat = convert(label)
                        labels.append(cat)
                    else:
                        print("Incorrect shape", shape)
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels)
    print("Images shape", images.shape)
    images = np.reshape(images, [images.shape[0],-1])
    images = images / 255
    print("Images shape", images.shape)
    print(images[0])
    labels = np.reshape(labels, labels.shape[0])
    labels = np.eye(9)[labels]
    print("Labels shape", labels.shape)
    return images, labels

def load_image(filename):
    raw = scipy.misc.imread(filename)
    shape = np.asarray(raw).shape
    if shape[0] != 180 and shape[1] != 240:
        print("Image shape is wrong", shape)
        exit(1)
    image = np.asarray(raw, dtype=np.float32) / 255
    image = np.reshape(image, -1)
    print("Unrolled shape", image.shape)
    return image

def shuffle_and_slice(images, labels, train_percentage=0.8):
    total_samples = images.shape[0]
    perm = numpy.arange(total_samples)
    numpy.random.shuffle(perm)
    shuffledXs = images[perm]
    shuffledYs = labels[perm]
    num_training_samples = int(total_samples * train_percentage)
    print("Number of training", num_training_samples)
    trainXs = shuffledXs[:num_training_samples]
    trainYs = shuffledYs[:num_training_samples]
    print("Number of test", total_samples - num_training_samples)
    testXs = shuffledXs[num_training_samples:]
    testYs = shuffledYs[num_training_samples:]
    print("Shape of test", testXs.shape)
    return trainXs, trainYs, testXs, testYs


def convert(label):
    categories = {'ski':0, 'epic':1, 'musical':2, 'extreme':3, 'pool':4, 'trump':5, 'nosignal':6, 'advertising':7,"sync":8}
    return categories[label]


def create_data_sets(fileName):
    """Create the data set for Into The Dungeon"""
    mat = scipy.io.loadmat(fileName)
    print("Keys ", mat.keys())
    ALLX = mat['ALLX']
    ALLY = mat['ALLY']
    print("Shape of ALLY", ALLY.shape)
    ALLY = np.reshape(ALLY, ALLY.shape[0])
    ALLY0Based = ALLY - 1
    yLabs = np.eye(9)[ALLY0Based]  # Convert a list of num 0..6 to a list of hot position array [0 0 0 1 0 0 0]
    totalSamples = ALLY.shape[0]

    print("First value NOT shuffled", yLabs[0])
    perm = numpy.arange(totalSamples)
    numpy.random.shuffle(perm)
    shuffledXs = ALLX[perm]
    shuffledYs = yLabs[perm]
    print("First value shuffled", shuffledYs[0])

    numTrainingSamples = int(totalSamples * 0.8)
    print("Number of training", numTrainingSamples)
    trainXs = shuffledXs[:numTrainingSamples]
    trainYs = shuffledYs[:numTrainingSamples]
    print("Number of test", totalSamples - numTrainingSamples)
    testXs = shuffledXs[numTrainingSamples:]
    testYs = shuffledYs[numTrainingSamples:]
    print("Shape of test", testXs.shape)
    return trainXs, trainYs, testXs, testYs

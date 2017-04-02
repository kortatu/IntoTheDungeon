import os
import sys
real_path = os.path.realpath(__file__)
base_dir = real_path[:real_path.rfind("/")]
sys.path.append(base_dir+'/..')
import tensorflow as tf
import common.perceptron as perceptron
import common.dataset as ds
import numpy

categories = ['ski', 'epic', 'musical', 'extreme', 'pool', 'trump', 'nosignal']
num_args = len(sys.argv)
if num_args < 2:
    print("Usage pyton evaluator.py image_file_name")
    exit(1)
image_file_name = sys.argv[1]
imageInput = ds.load_image(image_file_name)
shape = imageInput.shape
print("Shape", imageInput)
# Change this with loading only the image required
# filename = 'megaImages.mat'
# _, _, testXs, testYs = ds.create_data_sets(filename)
# imageInput = numpy.transpose(testXs[0])
# print("Image input: ", imageInput)
# label = testYs[0]
# print("The image to predict is supposed to be ", label)
print("Image input shape: ", imageInput.shape)
n_classes = 7
n_input = imageInput.shape[0]
print("Input features", n_input)
# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

weights, biases = perceptron.getVariables(n_input, n_classes)
pred = perceptron.multilayer_perceptron(x, weights, biases)
#Evaluator
evaluator = tf.nn.softmax(pred)
inputToEvaluate = [imageInput]
feed_dict = {x: inputToEvaluate}
# Initializing the variables
init = tf.global_variables_initializer()
model_path = base_dir + "/../saves/epicmodel.ckpt"
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, model_path)
    softmax = evaluator.eval(feed_dict)[0]
    print("Softmax:", softmax)
    maxCategoryIndex = numpy.argmax(softmax)
    print("Category predicted = ", categories[maxCategoryIndex])
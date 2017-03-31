import tensorflow as tf
import perceptron as perceptron
import dataset as ds
import numpy
import sys

categories = ['ski', 'epic', 'musical', 'extreme', 'pool', 'trump', 'nosignal']
num_args = len(sys.argv)
if num_args < 2:
    print("Usage pyton evaluator.py image_file_name")
    exit(1)
image_file_name = sys.argv[1]
print("Filename, ", image_file_name)
with tf.gfile.FastGFile(image_file_name, 'r') as f:
    image_data = f.read()
imageInput = tf.image.decode_jpeg(image_data)
print("Image decoded", imageInput)
imageInput = tf.cast(imageInput, tf.float32)
print("Image input 1-255", imageInput)
imageInput = numpy.multiply(imageInput, 1.0 / 255.0)
shape = tf.shape(imageInput)
print("Shape", shape)
imageInput = tf.reshape(imageInput, [shape[0] * shape[1] * shape[2]])
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
n_input = 129600
print("Input features", n_input)
# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

weights, biases = perceptron.getVariables(n_input, n_classes)
pred = perceptron.multilayer_perceptron(x, weights, biases)
# Initializing the variables
init = tf.global_variables_initializer()
model_path = "saves/truecolor/epicmodel.ckpt"
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, model_path)
    evaluator = tf.argmax(pred, 1)
    topPred = tf.nn.top_k(pred, 3)

    inputToEvaluate = [imageInput.eval()]
    print("Input to evaluate: " , inputToEvaluate)
    predictions = pred.eval({x: inputToEvaluate})
    print("Pred: ", predictions)
    maxCategoryIndex = evaluator.eval({x: inputToEvaluate})[0]
    print("Eval: ", maxCategoryIndex)
    values, indices = sess.run(topPred, {x: inputToEvaluate})
    i=1
    print("Top predictions: ", indices)
    for index in indices[0]:
        print(i, " - Prediction: ",categories[index])
        i = i + 1
    print("Category predicted = ", categories[maxCategoryIndex])
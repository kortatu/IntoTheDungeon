from flask import Flask
from flask import request
from flask import jsonify

import tensorflow as tf
import perceptron as perceptron
import urllib
import uuid
import numpy

app = Flask(__name__, static_url_path='')

categories = ['ski', 'epic', 'musical', 'extreme', 'pool', 'trump', 'nosignal']

def generate_filepath():
    unique_filename = str(uuid.uuid4())
    return "tmp/" + unique_filename + ".jpg"

def classify_image(image_path):
    with tf.gfile.FastGFile(image_path, 'r') as f:
        image_data = f.read()
    imageInput = tf.image.decode_jpeg(image_data)
    print("Image decoded", imageInput)
    imageInput = tf.cast(imageInput, tf.float32)
    print("Image input 1-255", imageInput)
    imageInput = numpy.multiply(imageInput, 1.0 / 255.0)
    shape = tf.shape(imageInput)
    print("Shape", shape)
    imageInput = tf.reshape(imageInput, [shape[0] * shape[1] * shape[2]])

    inputToEvaluate = [imageInput.eval(session = tf_session)]
    print("Input to evaluate: " , inputToEvaluate)
    predictions = pred.eval({x: inputToEvaluate}, session = tf_session)
    print("Pred: ", predictions)
    _, indices = tf_session.run(topPred, {x: inputToEvaluate})

    res = []    
    print("Top predictions: ", indices)
    for index in indices[0]:
        res.append(categories[index])

    return res

@app.route("/api/v1/classify")
def classify():
    if (request.args is None or request.args.get('url') is None):
        return ""

    filepath = generate_filepath()
    urllib.urlretrieve(request.args.get('url'), filepath)
    
    return jsonify(classify_image(filepath))

@app.route('/')
def root():
    return app.send_static_file('index.html')

n_classes = 7
n_input = 129600

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

weights, biases = perceptron.getVariables(n_input, n_classes)
pred = perceptron.multilayer_perceptron(x, weights, biases)

# Initializing the variables
init = tf.global_variables_initializer()
model_path = "../saves/truecolor/epicmodel.ckpt"
saver = tf.train.Saver()
tf_session = tf.Session()
tf_session.run(init)

saver.restore(tf_session, model_path)
evaluator = tf.argmax(pred, 1)
topPred = tf.nn.top_k(pred, 3)

if __name__ == "__main__":
    app.run()
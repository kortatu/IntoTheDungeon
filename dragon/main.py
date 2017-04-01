from flask import Flask
from flask import request
from flask import jsonify
import sys
sys.path.append('../common')
import tensorflow as tf
import perceptron as perceptron
import urllib
import uuid
import dataset as dataset

app = Flask(__name__, static_url_path='')

categories = ['ski', 'epic', 'musical', 'extreme', 'pool', 'trump', 'nosignal']

def generate_filepath():
    unique_filename = str(uuid.uuid4())
    return "tmp/" + unique_filename + ".jpg"

def classify_image(image_path):
    imageInput = dataset.load_image(image_path)
    inputToEvaluate = [imageInput]
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
model_path = "../saves/epicmodel.ckpt"
saver = tf.train.Saver()
tf_session = tf.Session()
tf_session.run(init)

saver.restore(tf_session, model_path)
evaluator = tf.argmax(pred, 1)
topPred = tf.nn.top_k(pred, 1)

if __name__ == "__main__":
    app.run(port=5001)
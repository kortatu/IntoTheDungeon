import sys
import os
real_path = os.path.realpath(__file__)
base_dir = real_path[:real_path.rfind("/")]
sys.path.append(base_dir+'/..')
from flask import Flask
from flask import request
from flask import jsonify
import tensorflow as tf
import common.perceptron as perceptron
import uuid
import common.dataset as dataset
import urllib2


app = Flask(__name__, static_url_path='')

categories = ['ski', 'epic', 'musical', 'extreme', 'pool', 'trump', 'nosignal', 'advertising', 'nosync', 'balloon']
authHeader = {"Authorization": "Basic ZGVtaXVyZ286ZmxleA=="}


def generate_filepath():
    unique_filename = str(uuid.uuid4())
    return base_dir + "/tmp/" + unique_filename + ".jpg"


def classify_image(image_path):
    values_list, indices_list = tf_session.run(topPred, {x: [dataset.load_image(image_path)]})
    res = []

    values = values_list[0]
    indices = indices_list[0]
    for i, item in enumerate(values):
        category = categories[indices[i]]
        res.append({"category": category, "score": str(item)})

    return res


@app.route("/api/v1/classify")
def classify():
    if request.args is None or request.args.get('url') is None:
        return ""

    filepath = generate_filepath()

    req = urllib2.Request(url=request.args.get('url'), headers=authHeader)
    with open(filepath,'wb') as f:
        f.write(urllib2.urlopen(req).read())
    return jsonify(classify_image(filepath))


@app.route('/')
def root():
    return app.send_static_file('index.html')


n_classes = 10
n_input = 129600

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

weights, biases = perceptron.getVariables(n_input, n_classes)
pred = perceptron.multilayer_perceptron(x, weights, biases)

# Initializing the variables
init = tf.global_variables_initializer()
model_path = base_dir + "/../saves/epicmodel.ckpt"
saver = tf.train.Saver()
tf_session = tf.Session()
tf_session.run(init)

saver.restore(tf_session, model_path)
evaluator = tf.nn.softmax(pred)
topPred = tf.nn.top_k(evaluator, 3)

if __name__ == "__main__":
    app.run(host= '0.0.0.0', port=5001)
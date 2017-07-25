import sys
import os
import uuid
import tensorflow as tf
import time
real_path = os.path.realpath(__file__)
base_dir = real_path[:real_path.rfind("/")]
sys.path.append(base_dir+'/..')
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template, send_from_directory

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Supress tensorflow compilation warnings

app = Flask(__name__)

app.config['video'] = 'video2.mp4'

sess = tf.Session()
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile(base_dir + "/retrained_labels.txt")]

with tf.gfile.FastGFile( base_dir + "/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

@app.route('/')
def index():
    return render_template('index.html', config=app.config)


def generate_filepath():
    unique_filename = str(uuid.uuid4())
    return base_dir + "/tmp/" + unique_filename + ".jpeg"


def classify_image(image_path):
    # Loads label file, strips off carriage return
    res = []

    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
                            {'DecodeJpeg/contents:0': image_data})


    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    top_score = -1  # in order to make sure top_score changes its value
    top_category = ""

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]

        if score > top_score:
            top_score = round(score, 3)
            top_category = human_string



    res.append({"topCategory": top_category, "score": top_score})
    return res


@app.route("/api/v1/classify", methods=['POST'])
def classify():
    data = request.get_json()

    if data is None or data.get('imgBase64') is None:
        return ""

    filepath = generate_filepath()

    with open(filepath, "wb") as fh:
        imgData = data.get('imgBase64')
        missing_padding = len(imgData) % 4
        if missing_padding != 0:
            imgData += b'='* (4 - missing_padding)
        fh.write(imgData.decode('base64'))

    result = classify_image(filepath)

    os.remove(filepath)
    return jsonify(result[0])


@app.route('/static/css/<path:path>')
def send_css(path):
    return send_from_directory('static/css', path)


@app.route('/static/js/<path:path>')
def send_js(path):
    return send_from_directory('static/js', path)


@app.route('/static/videos/<path:path>')
def send_video(path):
    return send_from_directory('static/video', path)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6006, threaded=True)
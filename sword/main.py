import sys
import os
import uuid
real_path = os.path.realpath(__file__)
base_dir = real_path[:real_path.rfind("/")]
sys.path.append(base_dir+'/..')
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template, make_response, send_file, send_from_directory

app = Flask(__name__)

app.config['video'] = 'bunny.mp4'
app.config['categories'] = ['smoker', 'nonsmoker']


@app.route('/')
def index():
    return render_template('index.html', config = app.config)


def generate_filepath():
    unique_filename = str(uuid.uuid4())
    return base_dir + "/tmp/" + unique_filename + ".jpg"


def classify_image(image_path):
    res = []

    res.append({"category": "test", "score": "0.1"})

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

    return jsonify(result)


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
    app.run(host='0.0.0.0', port=5000, threaded=True)
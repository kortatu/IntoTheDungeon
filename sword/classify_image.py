
import sys
import os
import numpy as np
import uuid
import tensorflow as tf
import time
from scipy import misc
real_path = os.path.realpath(__file__)
base_dir = real_path[:real_path.rfind("/")]
sys.path.append(base_dir+'/..')
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template, send_from_directory
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2
import time

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Supress tensorflow compilation warnings


def classify_image(image_path):
    # Loads label file, strips off carriage return
    res = []

    # image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    image_np = misc.imread(image_path)
    # # Feed the image_data as input to the graph and get first prediction
    # softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    # predictions = sess.run(softmax_tensor, \
    #                         {'DecodeJpeg/contents:0': image_data})


    # # Sort to show labels of first prediction in order of confidence
    # top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    # top_score = -1  # in order to make sure top_score changes its value
    # top_category = ""

    # for node_id in top_k:
    #     human_string = label_lines[node_id]
    #     score = predictions[0][node_id]

    #     if score > top_score:
    #         top_score = round(score, 3)
    #         top_category = human_string

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')    
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    print("Before Run boxes, scores, num_detections", num_detections)
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    print("Run boxes, scores, num_detections", num_detections)

    print( np.squeeze(boxes),
         np.squeeze(classes).astype(np.int32),
         np.squeeze(scores))
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.031,
        line_thickness=8)

    misc.imsave('./tmp-output/' + str(time.time()) + 'outfile.jpg', image_np)

    res.append({"topCategory": "1", "score": "1"})
    return res

#Object detection initialize
CWD_PATH = os.getcwd()
# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'first_model'
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, 'object-detection.pbtxt')
NUM_CLASSES = 2
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
    print('Session loaded', sess)

print("Arg" , sys.argv[1])
classify_image(sys.argv[1])
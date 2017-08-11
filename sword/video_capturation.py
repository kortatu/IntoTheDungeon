import os
import argparse
import re
import tensorflow as tf
import sys
from shutil import copyfile


def number_key(name):
   parts = re.findall('[^0-9]+|[0-9]+', name)
   L = []
   for part in parts:
       try:
          L.append(int(part))
       except ValueError:
          L.append(part)
   return L

real_path = os.path.realpath(__file__)
real_path = real_path[0:real_path.rfind('\\')] #in linux change \\ for /

parser = argparse.ArgumentParser(description='Detect smokers in a video')
parser.add_argument('-i', '--input', help='Input video')
parser.add_argument('-o', '--output', default='output/', help='directory where images will be classified')
parser.add_argument('-f', '--fps', default='1', help='Number of fps to analyze')
parser.add_argument('-c', '--classifier', default=real_path)
args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)

print('Args:')
print(args)

print('ffmpeg command:')
print("ffmpeg -i " + args.input + " -vf fps=" + args.fps + " -q:v 1 " + args.output + "%d.jpeg")

os.system("ffmpeg -i " + args.input + " -vf fps=" + args.fps + " -q:v 1 " + args.output + "%d.jpeg")

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                in tf.gfile.GFile("/tf_files/retrained_labels.txt")]

    # Unpersists graph from file
with tf.gfile.FastGFile("/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

print(sorted(os.listdir(args.output), key=number_key))
log_file = open(args.input + "_log_file.txt", "w")


for image in sorted(os.listdir(args.output), key=number_key):
    log_file.write(image + '\n')
    print("Checking for image: " + args.output + image)
    image_path = args.output + image

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                               {'DecodeJpeg/contents:0': image_data})


        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        top_score = -1 # in order to make sure top_score changes its value
        top_category = ""

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if score > top_score:
                top_score = score
                top_category = human_string
            print('%s (score = %.5f)' % (human_string, score))
            log_file.write('%s (score = %.5f)' % (human_string, score) + '\n')
        print("It was probably a %s since p = %0.5f" % (top_category, top_score))
        log_file.write("It was probably a %s since p = %0.5f" % (top_category, top_score) + '\n')

        if not os.path.exists(top_category):
            os.makedirs(top_category)

        copyfile(image_path, top_category + '/' + top_category + image)



    sess.close()
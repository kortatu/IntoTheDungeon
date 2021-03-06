# encoding: utf-8
'''
Author: Alvaro González
Project: https://github.com/kortatu/IntoTheDungeon
'''

from __future__ import print_function
import os
import sys
real_path = os.path.realpath(__file__)
base_dir = real_path[:real_path.rfind("/")]
sys.path.append(base_dir+'/..')
import tensorflow as tf
import common.dataset as ds
import common.perceptron as perceptron

images, labels = ds.load_dirs(base_dir + "/trainImages")
trainXs, trainYs, testXs, testYs = ds.shuffle_and_slice(images, labels)

# filename = 'megaImages_gray_s.mat'
# filename = 'megaImages.mat'
# trainXs, trainYs, testXs, testYs = ds.create_data_sets(filename)
dataset = ds.DataSet(trainXs, trainYs, reshape=False)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)  # 0.333
# Parameters
learning_rate = 0.001
miLambda = 0.04
training_epochs = 275
batch_size = 75
display_step = 5
train_accuracy_step = 5
test_accuracy_step = 2

# Network Parameters
n_hidden_1 = 2000          # 1st layer number of features
n_hidden_2 = 256           # 2nd layer number of features
n_input = trainXs.shape[1]    # Images data input (img shape: 180*240)
n_classes = trainYs.shape[1]  # total classes (1-x categories)

# tf Graph input
x = tf.placeholder("float", [None, n_input], name="x")
y = tf.placeholder("float", [None, n_classes], name="labels")
la = tf.constant(miLambda, "float", )

# Construct model
weights, biases = perceptron.getVariables(n_input, n_classes)
pred = perceptron.multilayer_perceptron(x, weights, biases)
regul = la * (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['out']))

# Define loss and optimizer
with tf.name_scope("xentropy"):
    sce = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
    mean_entropy = tf.reduce_mean(sce, name="mean_entropy")
    tf.summary.scalar('cross_entropy', mean_entropy)
    tf.summary.scalar('regularization', regul)
    cost = mean_entropy + regul
    tf.summary.scalar('cost', cost)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
saver = tf.train.Saver()
model_path = base_dir + "/latest/epicmodel.ckpt"
with tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)) as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("/tmp/mapmaker/1")
    merge_summary = tf.summary.merge_all()
    writer.add_graph(sess.graph)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('accuracy', accuracy)

    # print("Shuffled? " , train_batch)
    print("Starting optimization")
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = trainXs.shape[0] / batch_size
        # Loop over batches
        for i in range(total_batch):
            batch_x, batch_y = dataset.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            print("Epoch:", '%04d' % (epoch + 1), "Batch:", '%02d' % (i + 1), "/", '%02d' % total_batch, "cost=", "{:.9f}".format(c))
            s = sess.run(merge_summary, feed_dict={x: batch_x, y: batch_y})
            summary_index = epoch * total_batch + i

            avg_cost += c / total_batch

        print("Epoch:", '%04d' % (epoch + 1), "Average cost=", "{:.9f}".format(avg_cost))
        if summary_index % display_step == 0:
            print("Summary index:", summary_index)
            writer.add_summary(s, summary_index)
        if (epoch + 1) % test_accuracy_step == 0:
            aTestResult = sess.run(pred, feed_dict={x: testXs[0:1]})
            print("A test result", aTestResult, "ITs label:", testYs[0:1])
            print("Epoch:", '%04d' % (epoch + 1), "Accuracy test:", accuracy.eval({x: testXs, y: testYs}))

    # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

    print("Accuracy train:", accuracy.eval({x: trainXs, y: trainYs}))
    print("Accuracy test:", accuracy.eval({x: testXs, y: testYs}))


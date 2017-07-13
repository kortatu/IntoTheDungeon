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
sys.path.append(base_dir + '/..')
import tensorflow as tf
import common.dataset as ds
import common.cnn as cnn
import argparse

parser = argparse.ArgumentParser(description="Train cnn or evaluate on test set")
parser.add_argument('-t', '--test', default=None,
                    help='Execute accuracy on test set loaded from directory. Default test/')
parser.add_argument('-r', '--restart', action='store_const', const=True,
                    help='Restart training')
parser.add_argument('-c', '--cost', default=900,
                    help='Minimum cost to save')

args = parser.parse_args()
print("Args", args)

paths, labels = ds.load_dirs_with_labels("/tmp/smoke_images")
# load_dirs(base_dir + "/trainImages")
# trainXs, trainYs, testXs, testYs = ds.shuffle_and_slice(paths, labels)

dataset = ds.PathDataSet(paths, labels, 0.9)

# test_xs, test_ys = dataset.test_images_and_labels(max=100)
# print("test_ys", test_ys)
# exit(0)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)  # 0.333
# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 20
display_step = 5
train_accuracy_step = 2
test_accuracy_step = 2

# Network Parameters
# n_input = trainXs.shape[1]    # Images data input (img shape: 180*240)
# TODO: extract n_input from datasource from sample image
n_input = 300*400*3    # Images data input (img shape: h*v*deep)
n_classes = labels.shape[1]  # total classes (1-x categories)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input], name="x")
y = tf.placeholder(tf.float32, [None, n_classes], name="labels")
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Construct model
weights, biases = cnn.get_variables(n_input, n_classes)
# Construct model
pred = cnn.conv_net(300, 400, x, weights, biases, keep_prob)


# Define loss and optimizer
softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(softmax_cross_entropy, name="mean_entropy")
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
saver = tf.train.Saver()
model_path = base_dir + "/latest/epicmodelcnn.ckpt"

with tf.Session(config=(tf.ConfigProto())) as sess:
    sess.run(init)
    if args.restart is None:
        saver.restore(sess, model_path)
    if args.test is None:
        writer = tf.summary.FileWriter("/tmp/mapmaker/2")
        merge_summary = tf.summary.merge_all()
        writer.add_graph(sess.graph)
        tf.summary.scalar('accuracy', accuracy)

        # print("Shuffled? " , train_batch)
        print("Starting optimization")
        best_cost = args.cost
        for epoch in range(training_epochs):
            avg_cost = 0.
            avg_accu = 0.
            total_batch = dataset.number_of_batches(batch_size)
            # Loop over batches
            for i in range(total_batch):
                batch_x, batch_y = dataset.next_batch(batch_size)
                _, c, accuracy_t = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
                print("Epoch:", '%04d' % (epoch + 1), "Batch:", '%02d' % (i + 1), "/", '%02d' % total_batch,
                      "cost=", "{:.4f}".format(c), "Accuracy train:", accuracy_t)
                # s = sess.run(merge_summary, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                # summary_index = epoch * total_batch + i
                avg_cost += c / total_batch
                avg_accu += accuracy_t / total_batch

            print("Epoch:", '%04d' % (epoch + 1), "Average cost=", "{:.9f}".format(avg_cost))
            print("Epoch:", '%04d' % (epoch + 1), "Average accu=", "{:.9f}".format(avg_accu))

            if best_cost is None or best_cost > avg_cost:
                print("Best cost updated!")
                best_cost = avg_cost
                save_path = saver.save(sess, model_path)

            if (epoch + 1) % test_accuracy_step == 0:
                test_xs, test_ys, _  = dataset.test_images_and_labels(max=20)
                accuracy_value, test_cost = sess.run([accuracy, cost],
                                                     feed_dict={x: test_xs, y: test_ys, keep_prob: 1.})
                print("Epoch:", '%04d' % (epoch + 1), "Accuracy test:", accuracy_value, "| Test Cost:", test_cost)

        # Test model
        # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # # Calculate accuracy
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

    else:
        print("Loading tests from", args.test)
        test_paths, test_labels = ds.load_dirs_with_labels(args.test)
        test_dataset = ds.PathDataSet(test_paths, test_labels, 0)
        test_xs, test_ys, test_paths  = test_dataset.test_images_and_labels(max=200)
        accuracy_value, test_cost, sce, pred_v = sess.run([accuracy, cost, softmax_cross_entropy, pred],
                                                          feed_dict={x: test_xs, y: test_ys, keep_prob: 1.})
        print("Accuracy test:", accuracy_value, "| Test Cost:", test_cost, "|SCE: ", sce)
        print("Pred", pred_v, "| Label:", test_ys)
        print("Paths", test_paths)

        # print("Accuracy train:", accuracy.eval({x: trainXs, y: trainYs, keep_prob: 1.}))
        # print("Accuracy test:", accuracy.eval({x: testXs, y: testYs, keep_prob: 1.}))

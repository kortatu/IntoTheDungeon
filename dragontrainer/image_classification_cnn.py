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
import numpy as np

parser = argparse.ArgumentParser(description="Train cnn or evaluate on test set")
parser.add_argument('-t', '--test', default=None,
                    help='Execute accuracy on test set loaded from directory. Default test/')
parser.add_argument('-r', '--restart', action='store_const', const=True,
                    help='Restart training')
parser.add_argument('-c', '--cost', default=None,
                    help='Minimum cost to save')

args = parser.parse_args()
print("Args", args)

if args.test is None:
    paths, labels = ds.load_dirs_with_labels("smoke_images/training")
    test_paths, test_labels = ds.load_dirs_with_labels("smoke_images/test")
else:
    paths, labels = ds.load_dirs_with_labels(args.test)
    test_paths, test_labels = paths, labels
# load_dirs(base_dir + "/trainImages")
# trainXs, trainYs, testXs, testYs = ds.shuffle_and_slice(paths, labels)

dataset = ds.PathDataSet(paths, labels)
test_dataset = ds.PathDataSet(test_paths, test_labels)

# test_xs, test_ys = dataset.test_images_and_labels(max=100)
# print("test_ys", test_ys)
# exit(0)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) 
# Parameters
learning_rate = 0.003
# training_epochs = 1000
training_epochs = 500
#batch_size = 80
batch_size = 50
#batch_size = 1
display_step = 5
train_accuracy_step = 1
test_accuracy_step = 1

# Network Parameters
# n_input = trainXs.shape[1]    # Images data input (img shape: 180*240)
an_image = dataset.load_sample_image()
sample_shape = np.asarray(an_image).shape
n_input = sample_shape[0]*sample_shape[1]*sample_shape[2]    # Images data input (img shape: h*v*deep)
print ("Sample shape", sample_shape, "n_input", n_input)
if len(labels.shape) == 2:
    n_classes = labels.shape[1]  # total classes (1-x categories)
else:
    n_classes = 1
dropout = 0.5  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input], name="x")
y = tf.placeholder(tf.float32, [None, n_classes], name="labels")
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# Construct model
# Conv net with 11 layers
#cnn.get_variables
# Conv net with 7 layers
#cnn.get_variables_shallow
# Conv net for 336 image size
#cnn.get_variables_22223
get_variables = cnn.get_variables_shallow
weights, biases = get_variables(sample_shape, n_classes)
# Conv net with 11 layers
#cnn.conv_net
# Conv net with 7 layers size 224
#cnn.conv_net_shallow_22222(sample_shape[0], sample_shape[1], x, weights, biases, keep_prob)
# Conv net for 336 image size
#cnn.conv_net_22223(sample_shape[0], sample_shape[1], x, weights, biases, keep_prob)
get_conv_net = cnn.conv_net_shallow_22222
logits = get_conv_net(sample_shape[0], sample_shape[1], x, weights, biases, keep_prob)


# Define loss and optimizer
if n_classes > 1:
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
else:
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)

cost = tf.reduce_mean(cross_entropy, name = "mean_entropy")
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.1).minimize(cost)

threshold = 0.75
# Evaluate model
if n_classes > 1:
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
else:
    sigmoid = tf.sigmoid(logits)
    predicted_class = tf.greater(sigmoid, threshold)
    correct_pred = tf.equal(predicted_class, tf.equal(y, 1.0))

accuracy = tf.reduce_mean( tf.cast(correct_pred, tf.float32) )

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
saver = tf.train.Saver()
model_path = base_dir + "/latest/epicmodelcnn.ckpt"


def accuracy_test(test_dataset):
    total_batch = test_dataset.number_of_batches(batch_size)
    test_avg_cost = 0.
    test_avg_accu = 0.
    for i in range(total_batch):
        batch_x, batch_y, batch_paths = test_dataset.next_batch(batch_size)
        accuracy_value, test_cost, test_sigmoid = sess.run([accuracy, cost, sigmoid], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
        if batch_size == 1:
            print("Accuracy test batch:", accuracy_value, "| Cost test:", test_cost, "| Labels: ", batch_y, "| Sigmoid:", test_sigmoid, )
        else:
            print("Accuracy test batch:", accuracy_value, "| Cost test:", test_cost)            
        test_avg_accu += accuracy_value / total_batch
        test_avg_cost += test_cost / total_batch
    print("Average accuracy test :", test_avg_accu, "| Average cost test:", test_avg_cost)


def accuracy_test_step(test_dataset, epoch):
    test_total_batch = test_dataset.number_of_batches(batch_size)
    test_avg_cost = 0.
    test_avg_accu = 0.
    for i in range(test_total_batch):
        test_batch_x, test_batch_y, _ = test_dataset.next_batch(batch_size)
        accuracy_value, test_cost = sess.run([accuracy, cost], feed_dict={x: test_batch_x, y: test_batch_y, keep_prob: 1.})
        test_avg_accu += accuracy_value / test_total_batch
        test_avg_cost += test_cost / test_total_batch
    print("Epoch:", '%04d' % (epoch + 1), "Accuracy test:", "{:.9f}".format(test_avg_accu),
          "| Cost test:", "{:.9f}".format(test_avg_cost))


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

        best_cost = float(args.cost)
        print("Starting optimization, bestcost is ", best_cost)
        for epoch in range(training_epochs):
            avg_cost = 0.
            avg_accu = 0.
            total_batch = dataset.number_of_batches(batch_size)
            # Loop over batches
            for i in range(total_batch):
                batch_x, batch_y, batch_paths = dataset.next_batch(batch_size)
                _, c, accuracy_t = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
                # print("Epoch:", '%04d' % (epoch + 1), "Batch:", '%02d' % (i + 1), "/", '%02d' % total_batch,
                #       "cost=", "{:.4f}".format(c), "Accuracy train:", accuracy_t)

                # print("Lr_loss", "{:.4f}".format(lr_loss))
                # print("Y", y_r, "Y_hat", y_hat, "Sigmoid", sigmoid_r, "FT", ft)
                # s = sess.run(merge_summary, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                # summary_index = epoch * total_batch + i
                avg_cost += c / total_batch
                avg_accu += accuracy_t / total_batch

            print("Epoch:", '%04d' % (epoch + 1), "Accuracy training:", "{:.9f}".format(avg_accu),
                  "| Cost training:", "{:.9f}".format(avg_cost))
            # print("Epoch:", '%04d' % (epoch + 1), "Average cost=", "{:.9f}".format(avg_cost))
            # print("Epoch:", '%04d' % (epoch + 1), "Average accu=", "{:.9f}".format(avg_accu))

            if best_cost is None or best_cost > avg_cost:
                print("Best cost updated!")
                best_cost = avg_cost
                save_path = saver.save(sess, model_path)
            else:
                print("Previous best cost", best_cost, "was better, not updating")

            if (epoch + 1) % test_accuracy_step == 0:
                accuracy_test_step(test_dataset, epoch)
                # test_xs, test_ys, _  = dataset.test_images_and_labels(max=20)
                # accuracy_value, test_cost = sess.run([accuracy, cost],
                #                                      feed_dict={x: test_xs, y: test_ys, keep_prob: 1.})
                # print("Epoch:", '%04d' % (epoch + 1), "Accuracy test:", accuracy_value, "| Test Cost:", test_cost)
            print()

        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

    else:
        print("Loading tests from", args.test)
        accuracy_test(dataset)
        #
        #
        #
        # test_xs, test_ys, test_paths  = test_dataset.test_images_and_labels(max=200)
        # accuracy_value, test_cost, sce, pred_v = sess.run([accuracy, cost, softmax_cross_entropy, pred],
        #                                                   feed_dict={x: test_xs, y: test_ys, keep_prob: 1.})
        # print("Accuracy test:", accuracy_value, "| Test Cost:", test_cost, "|SCE: ", sce)
        # print("Pred", pred_v, "| Label:", test_ys)
        # print("Paths", test_paths)

        # print("Accuracy train:", accuracy.eval({x: trainXs, y: trainYs, keep_prob: 1.}))
        # print("Accuracy test:", accuracy.eval({x: testXs, y: testYs, keep_prob: 1.}))

'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import scipy.io
import tensorflow as tf
import numpy as np
import dataset as ds

# filename = 'megaImages_gray_s.mat'
filename = 'megaImages.mat'
trainXs, trainYs, testXs, testYs = ds.create_data_sets(filename)

# images, labels = tf.train.shuffle_batch([ALLX, yLabs], batch_size=trainingSamples,
#                                         capacity=trainingSamples, min_after_dequeue=100)

# Parameters
#learning_rate = 0.001
learning_rate = 0.001
#miLambda = 0.001
miLambda = 0.0004
#training_epochs = 30
training_epochs = 45
batch_size = 150
display_step = 1
train_accuracy_step = 5
test_accuracy_step = 5

# Network Parameters
n_hidden_1 = 2000          # 1st layer number of features
n_hidden_2 = 256           # 2nd layer number of features
n_input = trainXs.shape[1]    # MNIST data input (img shape: 28*28)
n_classes = 7              # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
la = tf.constant(miLambda, "float", )

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)
regul = la * (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['out']))

    # (la / n_input + n_hidden_1 + n_hidden_2) * tf.reduce_sum(tf.square(weights['h1'])) + tf.reduce_sum(tf.square(weights['h2'])) + tf.square(tf.reduce_sum(weights['out']))
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) + regul
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
saver = tf.train.Saver()
model_path = "/tmp/epicmodel.ckpt"
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    # for epoch in range(training_epochs):
    #     avg_cost = 0.
    #     total_batch = int(mnist.train.num_examples/batch_size)
    #     # Loop over all batches
    #     for i in range(total_batch):
    #         batch_x, batch_y = mnist.train.next_batch(batch_size)
    #         # Run optimization op (backprop) and cost op (to get loss value)
    #         _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
    #                                                       y: batch_y})
    #         # Compute average loss
    #         avg_cost += c / total_batch
    #     # Display logs per epoch step
    #     if epoch % display_step == 0:
    #         print("Epoch:", '%04d' % (epoch+1), "cost=", \
    #             "{:.9f}".format(avg_cost))
    # print("Optimization Finished!")

    # train_batch, train_labels = sess.run([images, labels])

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Shuffled? " , train_batch)
    print("Starting optimization")
    for epoch in range(training_epochs):
        avg_cost = 0.
        _, c = sess.run([optimizer, cost], feed_dict={x: trainXs, y: trainYs})
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch +1 ), "cost=", "{:.9f}".format(c))
        if (epoch +1 ) % train_accuracy_step == 0:
            print("Epoch:", '%04d' % (epoch +1 ), "Accuracy train:", accuracy.eval({x: trainXs, y: trainYs}))
        if (epoch +1 ) % test_accuracy_step == 0:
            print("Epoch:", '%04d' % (epoch +1 ), "Accuracy test:", accuracy.eval({x: testXs, y: testYs}))

    # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

    print("Accuracy train:", accuracy.eval({x: trainXs, y: trainYs}))
    print("Accuracy test:", accuracy.eval({x: testXs, y: testYs}))
    prediction = tf.argmax(y, 1)

    # print("Weights: ", sess.run([weights['h1'], weights['h2'], weights['out']]));

    # print(prediction.eval(feed_dict={x: ALLX[0:2]}))

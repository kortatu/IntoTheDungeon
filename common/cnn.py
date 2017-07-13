import tensorflow as tf


def get_variables(n_input, n_classes):
    # Store layers weight & bias
    first_layer_features = 16
    second_layer_features = 32
    third_layer_features = 64
    fourth_layer_features = 64
    fc_layer_features = 1024
    weights = {
        # 5x5 conv, 3 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 3, first_layer_features])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, first_layer_features, second_layer_features])),

        'wc3': tf.Variable(tf.random_normal([5, 5, second_layer_features, third_layer_features])),
        'wc4': tf.Variable(tf.random_normal([5, 5, third_layer_features, fourth_layer_features])),
        # fully connected, 3*4*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([15 * 20 * fourth_layer_features, fc_layer_features])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([fc_layer_features, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([first_layer_features])),
        'bc2': tf.Variable(tf.random_normal([second_layer_features])),
        'bc3': tf.Variable(tf.random_normal([third_layer_features])),
        'bc4': tf.Variable(tf.random_normal([fourth_layer_features])),
        'bd1': tf.Variable(tf.random_normal([fc_layer_features])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    return weights, biases


# Create model
def conv_net(image_x, image_y, x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, image_y, image_x, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2) # 150 * 200

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    # conv2 = maxpool2d(conv2, k=10)

    conv3 = maxpool2d(conv3, k=2) # 75 * 100

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv4 = maxpool2d(conv4, k=5) # 15 * 20

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    # return tf.sigmoid(out)
    return out

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
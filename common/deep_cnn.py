import tensorflow as tf


def get_variables(n_input, n_classes):
    # Store layers weight & bias
    first_layer_features = 16
    second_layer_features = 16
    third_layer_features = 16
    fourth_layer_features = 32
    fifth_layer_features = 64
    fc_layer_features = 1024
    last_x = 224 / 2 / 2 / 2 / 2 / 2
    last_y = 224 / 2 / 2 / 2 / 2 / 2
    weights = {
        # 5x5 conv, 3 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 3, first_layer_features])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, first_layer_features, second_layer_features])),

        'wc3': tf.Variable(tf.random_normal([5, 5, second_layer_features, third_layer_features])),
        'wc4': tf.Variable(tf.random_normal([5, 5, third_layer_features, fourth_layer_features])),
        'wc5': tf.Variable(tf.random_normal([5, 5, fourth_layer_features, fourth_layer_features])),
        'wc6': tf.Variable(tf.random_normal([5, 5, fourth_layer_features, fourth_layer_features])),
        'wc7': tf.Variable(tf.random_normal([5, 5, fourth_layer_features, fourth_layer_features])),
        'wc8': tf.Variable(tf.random_normal([5, 5, fourth_layer_features, fourth_layer_features])),
        'wc9': tf.Variable(tf.random_normal([5, 5, fourth_layer_features, fourth_layer_features])),
        'wc10': tf.Variable(tf.random_normal([5, 5, fourth_layer_features, fourth_layer_features])),
        'wc11': tf.Variable(tf.random_normal([5, 5, fourth_layer_features, fourth_layer_features])),
        # fully connected, 3*4*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([7*7 * fourth_layer_features, fc_layer_features])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([fc_layer_features, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([first_layer_features])),
        'bc2': tf.Variable(tf.random_normal([second_layer_features])),
        'bc3': tf.Variable(tf.random_normal([third_layer_features])),
        'bc4': tf.Variable(tf.random_normal([fourth_layer_features])),
        'bc5': tf.Variable(tf.random_normal([fourth_layer_features])),
        'bc6': tf.Variable(tf.random_normal([fourth_layer_features])),
        'bc7': tf.Variable(tf.random_normal([fourth_layer_features])),
        'bc8': tf.Variable(tf.random_normal([fourth_layer_features])),
        'bc9': tf.Variable(tf.random_normal([fourth_layer_features])),
        'bc10': tf.Variable(tf.random_normal([fourth_layer_features])),
        'bc11': tf.Variable(tf.random_normal([fourth_layer_features])),
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
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2) # 112 * 112

    # Convolution Layer
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])

    # Max Pooling (down-sampling)
    # conv2 = maxpool2d(conv2, k=10)

    conv5 = maxpool2d(conv5, k=2) # 56 * 56

    conv6 = conv2d(conv5, weights['wc6'], biases['bc6'])
    conv7 = conv2d(conv6, weights['wc7'], biases['bc7'])
    pool3 = maxpool2d(conv7, k=2) # 28 * 28

    conv8 = conv2d(pool3, weights['wc8'], biases['bc8'])
    conv9 = conv2d(conv8, weights['wc9'], biases['bc9'])
    pool4 = maxpool2d(conv9, k=2) # 14 * 14

    conv10 = conv2d(pool4, weights['wc10'], biases['bc10'])
    conv11 = conv2d(conv10, weights['wc11'], biases['bc11'])
    pool5 = maxpool2d(conv11, k=2) # 7 * 7
    # Fully connected layer
    # Reshape conv7 output to fit fully connected layer input
    fc1 = tf.reshape(pool5, [-1, weights['wd1'].get_shape().as_list()[0]])
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
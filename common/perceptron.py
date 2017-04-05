import tensorflow as tf

def getVariables(n_input, n_classes):
    # Network Parameters
    n_hidden_1 = 2000          # 1st layer number of features
    n_hidden_2 = 256           # 2nd layer number of features
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="w1"),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="w2"),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name="outW")
    }
    biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden_1]), name="b1"),
        'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden_2]), name="b2"),
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), name="outB")
    }
    with tf.name_scope('hidden1'):
        tf.summary.histogram("weights", weights['h1'])
        tf.summary.histogram("biases", biases['b1'])
    with tf.name_scope('hidden2'):
        tf.summary.histogram("weights", weights['h2'])
        tf.summary.histogram("biases", biases['b2'])
    with tf.name_scope('out'):
        tf.summary.histogram("weights", weights['out'])
        tf.summary.histogram("biases", biases['out'])



    return weights, biases

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    # layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    with tf.name_scope('hidden1'):
        tf.summary.histogram("activations", layer_1)
    with tf.name_scope('hidden2'):
        tf.summary.histogram("activations", layer_2)
    with tf.name_scope('out'):
        tf.summary.histogram("activations", out_layer)
    return out_layer
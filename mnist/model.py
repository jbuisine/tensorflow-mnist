import tensorflow as tf


# Softmax Regression Model
def regression(X):
    W = tf.Variable(tf.zeros([784, 10]), name="W")
    B = tf.Variable(tf.zeros([10]), name="B")
    Y = tf.nn.softmax(tf.matmul(X, W) + B)
    return Y, [W, B]


# Multilayer Convolutional Network
def convolutional(X, pkeep):

    # three convolutional layers with their channel counts, and a
    # fully connected layer (tha last layer has 10 softmax neurons)
    K = 4  # first convolutional layer output depth
    L = 8  # second convolutional layer output depth
    M = 12  # third convolutional layer
    N = 200  # fully connected layer

    W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
    B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
    W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
    B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
    W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
    B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

    W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
    B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
    W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
    B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

    # The model
    stride = 1  # output is 28x28
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    stride = 2  # output is 14x14
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    stride = 2  # output is 7x7
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

    # reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

    Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
    YY4 = tf.nn.dropout(Y4, pkeep)
    Ylogits = tf.matmul(YY4, W5) + B5
    Y = tf.nn.softmax(Ylogits)

    return Y, Ylogits, [W1, B1, W2, B2, W3, B3, W4, B4, W5, B5]

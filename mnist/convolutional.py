import os
import model
import math
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

print("Tensorflow version " + tf.__version__)

# model
with tf.variable_scope("convolutional"):
    # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
    pkeep = tf.placeholder(tf.float32)
    # variable learning rate
    lr = tf.placeholder(tf.float32)
    Y, Ylogits, variables = model.convolutional(X, pkeep)

# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])

# train
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

saver = tf.train.Saver(variables)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        batch_X, batch_Y = mnist.train.next_batch(100)

        # learning rate decay
        max_learning_rate = 0.003
        min_learning_rate = 0.0001
        decay_speed = 2000.0
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)

        a, c = sess.run([accuracy, cross_entropy], {X: batch_X, Y_: batch_Y, pkeep: 1.00})

        # Test part
        if i % 100 == 0:
            a, c = sess.run([accuracy, cross_entropy], {X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
            print(str(i) + ": ********* epoch " + str(i * 100 // mnist.train.images.shape[0] + 1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

        # the backpropagation training step
        sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: 0.75})

    print("Test result")
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0}))

    path = saver.save(sess, os.path.join(os.path.dirname(__file__), 'data', 'convolutional.ckpt'), write_meta_graph=False, write_state=False)
    print("Saved:", path)

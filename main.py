import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from mnist import model


X = tf.placeholder("float", [None, 784])
sess = tf.Session()

# restore trained data
with tf.variable_scope("regression"):
    Y1, variables = model.regression(X)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/regression.ckpt")


with tf.variable_scope("convolutional"):
    pkeep = tf.placeholder(tf.float32)
    Y2, Ylogits, variables = model.convolutional(X, pkeep)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional.ckpt")


def regression(input):
    return sess.run(Y1, feed_dict={X: input}).flatten().tolist()


def convolutional(input):
    return sess.run(Y2, feed_dict={X: input, pkeep: 1.0}).flatten().tolist()


# webapp
app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output1 = regression(input)
    output2 = convolutional(input)
    return jsonify(results=[output1, output2])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
import time

from mnist import model

X1 = tf.placeholder("float", [None, 784])
X2 = tf.placeholder(tf.float32, [None, 28, 28, 1])
sess = tf.Session()

# restore trained data
with tf.variable_scope("regression"):
    Y1, variables = model.regression(X1)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/regression.ckpt")
print("Regression model restored.")


with tf.variable_scope("convolutional"):
    pkeep = tf.placeholder(tf.float32)
    Y2, Ylogits, variables = model.convolutional(X2, pkeep)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional.ckpt")
print("Convolutional model restored.")


def regression(input):
    return sess.run(Y1, feed_dict={X1: input}).flatten().tolist()


def convolutional(input):
    return sess.run(Y2, feed_dict={X2: input, pkeep: 1.0}).flatten().tolist()


# webapp
app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():

    startReg = time.time()
    input1 = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output1 = regression(input1)
    endReg = time.time()

    startConv = time.time()
    input2 = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 28, 28, 1)
    output2 = convolutional(input2)
    endConv = time.time()

    diffRef = endReg - startReg
    diffConv = endConv - startConv

    return jsonify(results=[output1, output2], times=[diffRef, diffConv])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

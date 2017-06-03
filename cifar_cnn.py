import tensorflow as tf
import numpy as np
import os
import cnn_util
from cifar_data_manager import CifarDataManager

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_DIR = '/Users/gsanchez/tensorflow_book/tmp/data'
NUM_STEPS = 5000
MINIBATCH_SIZE = 100

cifar = CifarDataManager()

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

conv1 = cnn_util.conv_layer(x, shape=[5, 5, 3, 32])
conv1_pool = cnn_util.max_pool_2x2(conv1)

conv2 = cnn_util.conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = cnn_util.max_pool_2x2(conv2)
conv2_flat = tf.reshape(conv2_pool, [-1, 8 * 8 * 64])

full_1 = tf.nn.relu(cnn_util.full_layer(conv2_flat, 1024))
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = cnn_util.full_layer(full1_drop, 10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def test(sess):
    X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
    Y = cifar.test.labels.reshape(10, 1000, 10)
    acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0})
                   for i in range(10)])
    print "Accuracy: {:.4}%".format(acc * 100)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(NUM_STEPS):
        batch = cifar.train.next_batch(MINIBATCH_SIZE)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1],
                                                           keep_prob: 1.0})
            print "step {}, training accuracy {}".format(i, train_accuracy)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    test(sess)
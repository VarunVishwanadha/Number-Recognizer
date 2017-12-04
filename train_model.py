import argparse
import sys
import os
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import tensorlayer as tl

FLAGS = None
SIGMA = 6
ALPHA = 36

file_directory = os.getcwd()


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    training_batch_size = 100
    learning_rate = 0.3
    layer_nodes = [784, 800, 10]
    # layer_nodes = [784, 2500, 2000, 1500, 1000, 500, 10]
    w = []
    b = []
    y_list = []

    for idx in range(len(layer_nodes)-1):
        w.append(tf.get_variable('w'+str(idx), [layer_nodes[idx], layer_nodes[idx+1]], initializer=tf.random_normal_initializer(stddev=0.05)))
        b.append(tf.get_variable('b'+str(idx), [layer_nodes[idx+1]], initializer=tf.zeros_initializer()))
        if idx == 0:
            y_list.append(tf.nn.sigmoid(tf.matmul(x, w[idx]) + b[idx]))
        elif idx == len(layer_nodes)-2:
            y_list.append(tf.matmul(y_list[idx-1], w[idx]) + b[idx])
        else:
            y_list.append(tf.nn.sigmoid(tf.matmul(y_list[idx-1], w[idx]) + b[idx]))

    # Output
    y = y_list[len(y_list)-1]

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()

    # Train
    for idx in range(10000):
        print(idx)
        batch_xs, batch_ys = mnist.train.next_batch(training_batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # Elastic Transformed Images
        elastic_transformed_images = []
        for data in np.array(batch_xs):
            elastic_transformed_images.append(tl.prepro.elastic_transform(data.reshape((28, 28)), ALPHA, SIGMA))
        batch_elastic_transformed_xs = []
        for data in np.array(elastic_transformed_images):
            batch_elastic_transformed_xs.append(data.reshape(784,))
        sess.run(train_step, feed_dict={x: batch_elastic_transformed_xs, y_: batch_ys})

        # Normalized Images
        normalized_images = []
        for data in np.array(batch_xs):
            normalized_images.append(tl.prepro.samplewise_norm(data.reshape((28, 28, 1))))
        batch_normalized_xs = []
        for data in np.array(normalized_images):
            batch_normalized_xs.append(data.reshape(784,))
        sess.run(train_step, feed_dict={x: batch_normalized_xs, y_: batch_ys})

        # Noise Distorted Images
        noise_distorted_images = []
        for data in np.array(batch_xs):
            noise_distorted_images.append(tl.prepro.drop(data.reshape((28, 28)), keep=0.7))
        batch_noise_distorted_xs = []
        for data in np.array(noise_distorted_images):
            batch_noise_distorted_xs.append(data.reshape(784,))
        sess.run(train_step, feed_dict={x: batch_noise_distorted_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    save_path = saver.save(sess, file_directory+'/model.ckpt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

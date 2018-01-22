import tensorflow as tf
import numpy as np

def convnet(inputs, num_classes):
    conv1 = tf.layers.conv2d(inputs, 32, 4, 1, 'valid')
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, 'valid')

    conv2 = tf.layers.conv2d(pool1, 48, 5, 1, 'valid')
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, 'valid')

    conv3 = tf.layers.conv2d(pool2, 64, 5, 1, 'valid')
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2, 'valid')

    conv4 = tf.layers.conv2d(pool3, 128, 5, 1, 'valid')
    conv4 = tf.squeeze(conv4, [1, 2])

    fc5 = tf.layers.dense(conv4, 128, tf.nn.relu)
    logits = tf.layers.dense(fc5, num_classes)

    return logits

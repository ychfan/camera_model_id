import tensorflow as tf
import numpy as np

def convnet(inputs, num_classes, decay=0.0075):
    w_init = weight_init = tf.contrib.layers.xavier_initializer()
    w_reg = tf.contrib.layers.l2_regularizer(decay)

    conv1 = tf.layers.conv2d(inputs, 32, 4, 1, 'valid', kernel_initializer=w_init, kernel_regularizer=w_reg)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, 'valid')

    conv2 = tf.layers.conv2d(pool1, 48, 5, 1, 'valid', kernel_initializer=w_init, kernel_regularizer=w_reg)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, 'valid')

    conv3 = tf.layers.conv2d(pool2, 64, 5, 1, 'valid', kernel_initializer=w_init, kernel_regularizer=w_reg)
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2, 'valid')

    conv4 = tf.layers.conv2d(pool3, 128, 5, 1, 'same', kernel_initializer=w_init, kernel_regularizer=w_reg)
    conv4 = tf.reduce_mean(pool3, [1, 2])

    fc5 = tf.layers.dense(conv4, 128, tf.nn.relu, kernel_initializer=w_init, kernel_regularizer=w_reg)
    logits = tf.layers.dense(fc5, num_classes, kernel_initializer=w_init, kernel_regularizer=w_reg)

    return logits

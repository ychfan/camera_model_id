"""Simple convolutional neural network classififer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from common import metrics
from common import ops
from common import resnet


def get_params():
  return {
    "weight_decay": 0.0002,
  }


def model(features, labels, mode, params):
  """CNN classifier model."""
  images = features["image"]
  labels = labels["label"]

  training = mode == tf.estimator.ModeKeys.TRAIN

  x = images / 255.0

  def res_block(x, size, stride, training, name):
    with tf.variable_scope(name):
      x = tf.layers.batch_normalization(x, training=training)
      x = tf.nn.relu(x)
      x = tf.layers.conv2d(x, size / 4, 1, strides=stride, padding="same",
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))

      x = tf.layers.batch_normalization(x, training=training)
      x = tf.nn.relu(x)
      x = tf.layers.conv2d(x, size / 4, 3, padding="same",
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))

      x = tf.layers.batch_normalization(x, training=training)
      x = tf.nn.relu(x)
      x = tf.layers.conv2d(x, size, 1, padding="same",
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))
      return x

  def res_stage(x, input_channels, output_channels, stride, training, name):
    with tf.variable_scope(name):
      for block in range(3):
        x += res_block(x, input_channels, 1, training, "block" + str(block))
      res = res_block(x, output_channels, stride, training, "up")
      x = tf.layers.conv2d(x, output_channels, 1, strides=stride, padding="same",
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))
      x += res
      return x

  size = 4
  x = tf.layers.conv2d(x, size, 1, padding="same",
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))

  for stage in range(4):
    x = res_stage(x, size, size * 2, 4, training, "stage" + str(stage))
    size *= 2

  logits = tf.layers.dense(x, params.num_classes,
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))

  logits = tf.reduce_mean(logits, axis=[1, 2])

  predictions = tf.argmax(logits, axis=-1)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  eval_metrics = {
    "accuracy": tf.metrics.accuracy(labels, predictions),
    "top_1_error": tf.metrics.mean(metrics.top_k_error(labels, logits, 1)),
  }

  return {"predictions": predictions}, loss, eval_metrics

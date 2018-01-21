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
  x = tf.layers.conv2d(x, 4, 3, padding="same",
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))
  x = tf.layers.batch_normalization(x, training=training)
  x = tf.nn.relu(x)
  x = tf.layers.average_pooling2d(x, 8, 8, padding="same")
  x = tf.layers.conv2d(x, 4, 3, padding="same",
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))
  x = tf.layers.batch_normalization(x, training=training)
  x = tf.nn.relu(x)
  x = tf.layers.average_pooling2d(x, 8, 8, padding="same")

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

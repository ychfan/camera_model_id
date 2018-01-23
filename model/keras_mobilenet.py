"""Simple convolutional neural network classififer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common import metrics


def get_params():
  return {
    "weight_decay": 0.0002,
  }


def model(features, labels, mode, params):
  """CNN classifier model."""
  images = features["image"]
  labels = labels["label"]

  training = mode == tf.estimator.ModeKeys.TRAIN
  tf.keras.backend.set_learning_phase(training)

  images = images[:, 144:-144, 144:-144, :]
  images = tf.keras.applications.mobilenet.preprocess_input(images)
  mobilenet = tf.keras.applications.MobileNet(
    input_shape=(224, 224, 3), include_top=False,
    weights='imagenet' if training else None,
    input_tensor=images,
    pooling='avg')
  for layer in mobilenet.layers:
    layer.trainable = False

  logits = tf.layers.dense(mobilenet(images), params.num_classes,
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))

  predictions = tf.argmax(logits, axis=-1)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  eval_metrics = {
    "accuracy": tf.metrics.accuracy(labels, predictions),
    "top_1_error": tf.metrics.mean(metrics.top_k_error(labels, logits, 1)),
  }

  return {"predictions": predictions}, loss, eval_metrics

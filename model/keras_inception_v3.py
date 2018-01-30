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
  if mode != tf.estimator.ModeKeys.PREDICT:
    labels = labels["label"]

  training = mode == tf.estimator.ModeKeys.TRAIN
  tf.keras.backend.set_learning_phase(training)

  image_shape = (512, 512, 3)
  if training:
    image_shape = (140, 140, 3)
    images = tf.random_crop(images, [params.batch_size, 140, 140, 3])
  images = tf.keras.applications.inception_v3.preprocess_input(images)
  inception = tf.keras.applications.inception_v3.InceptionV3(
    input_shape=image_shape, include_top=False,
    weights='imagenet' if training else None,
    input_tensor=images,
    pooling='avg')
  for layer in inception.layers:
    layer.trainable = False

  logits = tf.layers.dense(inception(images), params.num_classes,
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))

  predictions = tf.argmax(logits, axis=-1)

  if mode == tf.estimator.ModeKeys.PREDICT:
    return {"predictions": predictions}, None, None

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  eval_metrics = {
    "accuracy": tf.metrics.accuracy(labels, predictions),
    "top_1_error": tf.metrics.mean(metrics.top_k_error(labels, logits, 1)),
  }

  return {"predictions": predictions}, loss, eval_metrics

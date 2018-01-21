"""Simple convolutional neural network classififer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

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

  slim = tf.contrib.slim
  vgg = nets.vgg
  with slim.arg_scope(vgg.vgg_arg_scope()):
    net, _ = vgg.vgg_16(images,
                        num_classes=params.num_classes,
                        is_training=training,
                        spatial_squeeze=False)

  logits = tf.reduce_mean(net, axis=[1, 2])

  predictions = tf.argmax(logits, axis=-1)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  eval_metrics = {
    "accuracy": tf.metrics.accuracy(labels, predictions),
    "top_1_error": tf.metrics.mean(metrics.top_k_error(labels, logits, 1)),
  }

  return {"predictions": predictions}, loss, eval_metrics

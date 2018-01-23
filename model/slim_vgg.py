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
    mean = tf.constant([123.68, 116.78, 103.94])
    images = images - mean
    net, _ = vgg.vgg_16(images,
                        num_classes=params.num_classes,
                        is_training=training,
                        spatial_squeeze=False)

    if training:
      variables_to_restore = slim.get_variables_to_restore(
        exclude=['vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8'])
      tf.train.init_from_checkpoint(
        "vgg_16.ckpt", {v.name.split(':')[0]: v for v in variables_to_restore})
      variables_in_fc = []
      for scope in ['vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8']:
        variables = tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_in_fc.extend(variables)
      variables_to_train = tf.get_collection_ref(
        tf.GraphKeys.TRAINABLE_VARIABLES)
      variables_to_train.clear()
      variables_to_train.extend(variables_in_fc)

  logits = tf.reduce_mean(net, axis=[1, 2])

  predictions = tf.argmax(logits, axis=-1)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  eval_metrics = {
    "accuracy": tf.metrics.accuracy(labels, predictions),
    "top_1_error": tf.metrics.mean(metrics.top_k_error(labels, logits, 1)),
  }

  return {"predictions": predictions}, loss, eval_metrics

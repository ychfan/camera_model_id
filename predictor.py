"""Predictor module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.client import device_lib

from common import hooks
from common import hparams
from common import io as common_io
from common import ops
from common import optimizer
from common import metrics

import dataset.camera
import dataset.camera_mem

import model.baseline
import model.keras_inception_v3
import model.keras_inception_resnet_v2
import model.slim_vgg

tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_string("model", "baseline", "Model name.")
tf.flags.DEFINE_string("dataset", "camera_mem", "Dataset name.")
tf.flags.DEFINE_string("output_dir", "", "Optional output dir.")
tf.flags.DEFINE_string("schedule", "train_and_evaluate", "Schedule.")
tf.flags.DEFINE_string("hparams", "", "Hyper parameters.")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training epochs.")
tf.flags.DEFINE_integer("shuffle_batches", 40, "Shuffle batches.")
tf.flags.DEFINE_integer("num_reader_threads", 8, "Num reader threads.")
tf.flags.DEFINE_integer("save_summary_steps", 1000, "Summary steps.")
tf.flags.DEFINE_integer("save_checkpoints_steps", 10000, "Checkpoint steps.")
tf.flags.DEFINE_integer("eval_frequency", 1, "Eval frequency.")
tf.flags.DEFINE_integer("num_gpus", 0, "Numner of gpus.")

FLAGS = tf.flags.FLAGS

MODELS = {
  "baseline": model.baseline,
  "keras_inception_v3": model.keras_inception_v3,
  "keras_inception_resnet_v2": model.keras_inception_resnet_v2,
  "slim_vgg": model.slim_vgg,
}

DATASETS = {
  "camera": dataset.camera,
  "camera_mem": dataset.camera_mem,
}


def main(unused_argv):
  if FLAGS.output_dir:
    model_dir = FLAGS.output_dir
  else:
    raise NotImplementedError

  DATASETS[FLAGS.dataset].prepare()

  session_config = tf.ConfigProto()
  session_config.allow_soft_placement = True
  session_config.gpu_options.allow_growth = True
  run_config = tf.contrib.learn.RunConfig(
    model_dir=model_dir,
    save_summary_steps=FLAGS.save_summary_steps,
    save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    save_checkpoints_secs=None,
    session_config=session_config)

  estimator = tf.estimator.Estimator(
    model_fn=optimizer.make_model_fn(
      MODELS[FLAGS.model].model, FLAGS.num_gpus),
    config=run_config,
    params=hparams.get_params(
      MODELS[FLAGS.model], DATASETS[FLAGS.dataset], FLAGS.hparams))

  y = estimator.predict(
    input_fn=common_io.make_input_fn(
      DATASETS[FLAGS.dataset],
      tf.estimator.ModeKeys.PREDICT,
      hparams.get_params(
        MODELS[FLAGS.model], DATASETS[FLAGS.dataset], FLAGS.hparams),
      num_epochs=1,
      shuffle_batches=False,
      num_threads=FLAGS.num_reader_threads),
  )

  print("fname,camera")
  classes = DATASETS[FLAGS.dataset].CLASSES
  for file, p in zip(DATASETS[FLAGS.dataset].TEST_LIST, y):
    print(file, classes[p["predictions"]], sep=',')


if __name__ == "__main__":
  tf.app.run()

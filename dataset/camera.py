"""Mnist dataset preprocessing and specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import random
import tensorflow as tf
import cv2

TRAIN_DIR = "data/train/"
VAL_LIST_FILE = "validation_list.txt"
TEST_DIR = "data/test/"
IMAGE_SIZE = 512
NUM_CLASSES = 12
CLASSES = ["HTC-1-M7", "iPhone-4s", "iPhone-6", "LG-Nexus-5x",
           "Motorola-Droid-Maxx", "Motorola-Nexus-6", "Motorola-X",
           "Samsung-Galaxy-Note3", "Samsung-Galaxy-S4", "Sony-NEX-7"]
TRAIN_LIST = []
VAL_LIST = []
TEST_LIST = []


def get_params():
  """Return dataset parameters."""
  return {
    "num_classes": len(CLASSES),
  }


def prepare():
  """This function will be called once to prepare the dataset."""
  TRAIN_LIST[:] = []
  VAL_LIST[:] = []
  TEST_LIST[:] = []
  with open(VAL_LIST_FILE) as f:
    VAL_LIST.extend(f.read().splitlines())
  for c in CLASSES:
    for file in os.listdir(TRAIN_DIR + c):
      filepath = c + "/" + file
      if filepath not in VAL_LIST:
        TRAIN_LIST.append(filepath)
  random.shuffle(TRAIN_LIST)
  for file in os.listdir(TEST_DIR):
    TEST_LIST.append(file)
  TEST_LIST.sort()


def read(mode):
  """Create an instance of the dataset object."""
  file_list = {
    tf.estimator.ModeKeys.TRAIN: TRAIN_LIST,
    tf.estimator.ModeKeys.EVAL: VAL_LIST,
    tf.estimator.ModeKeys.PREDICT: TEST_LIST,
  }[mode]

  label_list = []
  if mode == tf.estimator.ModeKeys.PREDICT:
    label_list = [0] * len(file_list)
  else:
    for file in file_list:
      c = file.split("/", 1)[0]
      idx = CLASSES.index(c)
      label_list.append(idx)

  return tf.contrib.data.Dataset.from_tensor_slices((file_list, label_list))


def parse(mode, filename, label):
  """Parse input record to features and labels."""
  def read_image(mode, filename):
    image = cv2.imread(TRAIN_DIR + filename.decode('utf-8'))
    if mode != tf.estimator.ModeKeys.PREDICT:
      if mode == tf.estimator.ModeKeys.TRAIN:
        manip = random.random() < 0.5
      else:
        manip = hash(filename) % 2 == 0
      if manip:
        if mode == tf.estimator.ModeKeys.TRAIN:
          manip = random.randrange(8)
        else:
          manip = hash(filename) % 8
        if manip == 0:  # JPEG compression with quality factor = 70
          encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
          _, image = cv2.imencode('.jpg', image, encode_param)
          image = cv2.imdecode(image, 1)
        elif manip == 1:  # JPEG compression with quality factor = 90
          encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
          _, image = cv2.imencode('.jpg', image, encode_param)
          image = cv2.imdecode(image, 1)
        elif manip == 2:  # resizing (via bicubic) by a factor of 0.5
          image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_CUBIC)
        elif manip == 3:  # resizing (via bicubic) by a factor of 0.8
          image = cv2.resize(image, (0, 0), fx=0.8, fy=0.8,
                             interpolation=cv2.INTER_CUBIC)
        elif manip == 4:  # resizing (via bicubic) by a factor of 1.5
          image = cv2.resize(image, (0, 0), fx=1.5, fy=1.5,
                             interpolation=cv2.INTER_CUBIC)
        elif manip == 5:  # resizing (via bicubic) by a factor of 2.0
          image = cv2.resize(image, (0, 0), fx=2.0, fy=2.0,
                             interpolation=cv2.INTER_CUBIC)
        elif manip == 6:  # gamma correction using gamma = 0.8
          image = image / 255.0
          image = cv2.pow(image, 0.8)
          image = image * 255.0
        elif manip == 7:  # gamma correction using gamma = 1.2
          image = image / 255.0
          image = cv2.pow(image, 1.2)
          image = image * 255.0
      # crop center patch
      width = np.size(image, 1)
      height = np.size(image, 0)
      left = (width - IMAGE_SIZE) / 2
      top = (height - IMAGE_SIZE) / 2
      if mode == tf.estimator.ModeKeys.TRAIN:
        left += random.gauss(0, 3)
        top += random.gauss(0, 3)
      left = round(left)
      top = round(top)
      image = image[top:top + IMAGE_SIZE, left:left + IMAGE_SIZE]
      image = image / 255.0
    return image.astype(np.float32)
  image = tf.py_func(read_image, [mode, filename], tf.float32, stateful=False)
  image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
  label = tf.to_int32(label)
  #label = tf.Print(label, [image, label])
  return {"image": image}, {"label": label}

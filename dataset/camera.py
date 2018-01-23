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
          image = jpeg_compression(image, 70)
        elif manip == 1:  # JPEG compression with quality factor = 90
          image = jpeg_compression(image, 90)
        elif manip == 2:  # resizing (via bicubic) by a factor of 0.5
          image = resize(image, 0.5)
        elif manip == 3:  # resizing (via bicubic) by a factor of 0.8
          image = resize(image, 0.8)
        elif manip == 4:  # resizing (via bicubic) by a factor of 1.5
          image = resize(image, 1.5)
        elif manip == 5:  # resizing (via bicubic) by a factor of 2.0
          image = resize(image, 2.0)
        elif manip == 6:  # gamma correction using gamma = 0.8
          image = gamma_correction(image, 0.8)
        elif manip == 7:  # gamma correction using gamma = 1.2
          image = gamma_correction(image, 1.2)
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
      if top + IMAGE_SIZE > height:
        top = height - IMAGE_SIZE
      if left + IMAGE_SIZE > width:
        left = width - IMAGE_SIZE
      if top < 0:
        top = 0
      if left < 0:
        left = 0
      image = image[top:top + IMAGE_SIZE, left:left + IMAGE_SIZE]
    # 'BGR'->'RGB'
    image = image[..., ::-1]
    return image.astype(np.float32)
  image = tf.py_func(read_image, [mode, filename], tf.float32, stateful=False)
  image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
  label = tf.to_int32(label)
  #label = tf.Print(label, [image, label])
  return {"image": image}, {"label": label}


def jpeg_compression(image, factor):
  encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), factor]
  _, image = cv2.imencode('.jpg', image, encode_param)
  return cv2.imdecode(image, 1)


def resize(image, factor):
  if np.size(image, 0) * factor >= IMAGE_SIZE and \
          np.size(image, 1) * factor >= IMAGE_SIZE:
    image = cv2.resize(image, (0, 0), fx=factor, fy=factor,
                       interpolation=cv2.INTER_CUBIC)
  return image


def gamma_correction(image, gamma):
  inv_gamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** inv_gamma) * 255
                    for i in np.arange(0, 256)]).astype("uint8")
  return cv2.LUT(image, table)
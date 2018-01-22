import numpy as np
import os
import random
import tensorflow as tf
import cv2

IMAGE_SIZE = 224

def data_loader(data_list, num_epochs):
    with open(data_list) as f:
        line_list = f.read().splitlines()
    filename_list = [_.split(' ')[0] for _ in line_list]
    label_list = [int(_.split(' ')[1]) for _ in line_list]

    filename_label_queue = tf.train.slice_input_producer([filename_list, label_list], num_epochs=num_epochs)
    return filename_label_queue[0], filename_label_queue[1]


def augment(filename, mode='train'):
  image = cv2.imread(filename.decode('utf-8'))
  manip = random.random() < 0.5
  manip = False  # TODO
  if manip:
    manip = random.randrange(8)
    if manip == 0:  # JPEG compression with quality factor = 70
      encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
      _, image = cv2.imencode('.jpg', image, encode_param)
      image = cv2.imdecode(image, 1)
    elif manip == 1:  # JPEG compression with quality factor = 90
      encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
      _, image = cv2.imencode('.jpg', image, encode_param)
      image = cv2.imdecode(image, 1)
    elif manip == 2:  # resizing (via bicubic) by a factor of 0.5
      width = np.size(image, 1)
      height = np.size(image, 0)
      if width * 0.5 >= IMAGE_SIZE and height * 0.5 >= IMAGE_SIZE:
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_CUBIC)
    elif manip == 3:  # resizing (via bicubic) by a factor of 0.8
      width = np.size(image, 1)
      height = np.size(image, 0)
      if width * 0.8 >= IMAGE_SIZE and height * 0.8 >= IMAGE_SIZE:
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
  try:
    width = np.size(image, 1)
  except AttributeError:
    print(filaname)
    print(type(image))
  height = np.size(image, 0)
  left = (width - IMAGE_SIZE) / 2
  top = (height - IMAGE_SIZE) / 2
  if mode == 'train':
    left += random.randint(0, 30)
    top += random.randint(0, 30)
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
  return image.astype(np.float32)

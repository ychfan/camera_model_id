import numpy as np
import os
import random
import tensorflow as tf
import cv2

IMAGE_SIZE = 64
IMAGE_MEAN = np.array([[[103.939, 116.779, 123.68]]])


def data_loader_aug(data_list, num_epochs):
    with open(data_list) as f:
        line_list = f.read().splitlines()
    filename_list = [_.split(' ')[0] for _ in line_list]
    label_list = [int(_.split(' ')[1]) for _ in line_list]

    filename_label_queue = tf.train.slice_input_producer([filename_list, label_list], num_epochs=num_epochs)

    image_file = tf.read_file(filename_label_queue[0])
    image = tf.image.decode_image(image_file, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    #image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])

    manip = tf.random_uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
    manip_type = tf.cond(tf.equal(manip, 0), lambda: tf.random_uniform(shape=(), minval=0, maxval=8, dtype=tf.int32), lambda: 8)

    image = tf.cond(tf.equal(manip, 0), lambda: tf.image.decode_jpeg(tf.image.encode_jpeg(image, quality=70), 3), lambda: image)
    image = tf.cond(tf.equal(manip, 0), lambda: tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3]), lambda: image)

    image = tf.cond(tf.equal(manip, 1), lambda: tf.image.decode_jpeg(tf.image.encode_jpeg(image, quality=90), 3), lambda: image)
    image = tf.cond(tf.equal(manip, 1), lambda: tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3]), lambda: image)

    image = tf.cond(tf.equal(manip, 2), lambda: tf.random_crop(image, [int(IMAGE_SIZE*0.5), int(IMAGE_SIZE*0.5), 3]), lambda: image)
    image = tf.cond(tf.equal(manip, 2), lambda: tf.squeeze(tf.image.resize_bicubic(tf.expand_dims(image, 0), [IMAGE_SIZE, IMAGE_SIZE]), [0]), lambda: tf.cast(image, tf.float32))

    image = tf.cond(tf.equal(manip, 3), lambda: tf.random_crop(image, [int(IMAGE_SIZE*0.8), int(IMAGE_SIZE*0.8), 3]), lambda: image)
    image = tf.cond(tf.equal(manip, 3), lambda: tf.squeeze(tf.image.resize_bicubic(tf.expand_dims(image, 0), [IMAGE_SIZE, IMAGE_SIZE]), [0]), lambda: tf.cast(image, tf.float32))

    image = tf.cond(tf.equal(manip, 4), lambda: tf.random_crop(image, [int(IMAGE_SIZE*1.5), int(IMAGE_SIZE*1.5), 3]), lambda: image)
    image = tf.cond(tf.equal(manip, 4), lambda: tf.squeeze(tf.image.resize_bicubic(tf.expand_dims(image, 0), [IMAGE_SIZE, IMAGE_SIZE]), [0]), lambda: tf.cast(image, tf.float32))

    image = tf.cond(tf.equal(manip, 5), lambda: tf.random_crop(image, [IMAGE_SIZE*2, IMAGE_SIZE*2, 3]), lambda: image)
    image = tf.cond(tf.equal(manip, 5), lambda: tf.squeeze(tf.image.resize_bicubic(tf.expand_dims(image, 0), [IMAGE_SIZE, IMAGE_SIZE]), [0]), lambda: tf.cast(image, tf.float32))

    image = tf.cond(tf.equal(manip, 6), lambda: tf.image.adjust_gamma(image, gamma=0.8), lambda:image)
    image = tf.cond(tf.equal(manip, 6), lambda: tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3]), lambda: image)

    image = tf.cond(tf.equal(manip, 7), lambda: tf.image.adjust_gamma(image, gamma=1.2), lambda:image)
    image = tf.cond(tf.equal(manip, 7), lambda: tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3]), lambda: image)

    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.cast(image, tf.float32)

    image = image - IMAGE_MEAN
    image = image * 0.0125

    return image, filename_label_queue[1]


def data_loader(data_list, num_epochs):
    with open(data_list) as f:
        line_list = f.read().splitlines()
    filename_list = [_.split(' ')[0] for _ in line_list]
    label_list = [int(_.split(' ')[1]) for _ in line_list]

    filename_label_queue = tf.train.slice_input_producer([filename_list, label_list], num_epochs=num_epochs)

    image_file = tf.read_file(filename_label_queue[0])
    image = tf.image.decode_image(image_file, channels=3)
    image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.cast(image, tf.float32)

    image = image - IMAGE_MEAN
    image = image * 0.0125

    return image, filename_label_queue[1]

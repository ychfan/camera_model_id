import numpy as np
import random
import cv2
import os

filelist = '/ifp/users/haichao/projects/camid/data/filelist/filelist_train'
save_path = '/ifp/users/haichao/projects/camid/data/train_aug'
min_image_size = 224

with open(filelist) as f:
    lines = f.read().splitlines()

count = 0
for line in lines:
    count += 1

    filepath = line.split(' ')[0]
    image = cv2.imread(filepath)
    folder = filepath.split('/')[-2]
    filename = filepath.split('/')[-1]
    save_folder = save_path + '/' + folder + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print('processing ' + str(count) + '-th image: ' + filename)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    _, image_aug = cv2.imencode('.png', image, encode_param)
    image_aug = cv2.imdecode(image_aug, 1)
    filename_aug = save_folder + filename[:-4] + '_qual70.png'
    cv2.imwrite(filename_aug, image_aug)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, image_aug = cv2.imencode('.png', image, encode_param)
    image_aug = cv2.imdecode(image_aug, 1)
    filename_aug = save_folder + filename[:-4] + '_qual90.png'
    cv2.imwrite(filename_aug, image_aug)

    width = np.size(image, 1)
    height = np.size(image, 0)

    if width * 0.5 >= min_image_size and height * 0.5 >= min_image_size:
        image_aug = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        filename_aug = save_folder + filename[:-4] + '_scale0.5.png'
        cv2.imwrite(filename_aug, image_aug)

    if width * 0.8 >= min_image_size and height * 0.8 >= min_image_size:
        image_aug = cv2.resize(image, (0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
        filename_aug = save_folder + filename[:-4] + '_scale0.8.png'
        cv2.imwrite(filename_aug, image_aug)

    image_aug = cv2.resize(image, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    filename_aug = save_folder + filename[:-4] + '_scale1.5.png'
    cv2.imwrite(filename_aug, image_aug)

    image_aug = cv2.resize(image, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    filename_aug = save_folder + filename[:-4] + '_scale2.0.png'
    cv2.imwrite(filename_aug, image_aug)

    image_aug = image / 255.0
    image_aug = cv2.pow(image_aug, 0.8)
    image_aug = image_aug * 255.0
    filename_aug = save_folder + filename[:-4] + '_pow0.8.png'
    cv2.imwrite(filename_aug, image_aug)

    image_aug = image / 255.0
    image_aug = cv2.pow(image_aug, 1.2)
    image_aug = image_aug * 255.0
    filename_aug = save_folder + filename[:-4] + '_pow1.2.png'
    cv2.imwrite(filename_aug, image_aug)

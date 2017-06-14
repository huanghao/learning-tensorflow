# coding: utf-8

import os
import glob

import tensorflow as tf
import numpy as np

import utils
from gen_captcha import gen_captcha_text_and_image
from const import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_next_batch(batch_size=DEFAULT_BATCH_SIZE):
    """生成一个训练batch"""
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = utils.convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128 mean为0
        batch_y[i, :] = utils.text2vec(text)

    return batch_x, batch_y


def transform_label(s):
    _, ext = os.path.splitext(str(s))
    text = os.path.basename(str(s))
    text = text.replace(ext, "")

    return utils.text2vec(text)


def read_inputs():
    files = glob.glob(TRAIN_SETS)
    filename_queue = tf.train.string_input_producer(files)
    reader = tf.WholeFileReader()

    key, val = reader.read(filename_queue)

    image = tf.image.decode_png(val, 3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_image_with_crop_or_pad(
        image, IMAGE_HEIGHT, IMAGE_WIDTH)

    label = tf.py_func(transform_label, [key], tf.float64)
    label = tf.reshape(label, [CHAR_SET_LEN * MAX_CAPTCHA])

    inputs = tf.train.shuffle_batch([image, label], batch_size=DEFAULT_BATCH_SIZE,
                                    capacity=DEFAULT_BATCH_SIZE * 5,
                                    min_after_dequeue=DEFAULT_BATCH_SIZE * 2)
    image_summary = tf.summary.image('image', inputs[0])

    return inputs, image_summary


if __name__ == "__main__":
    inputs, image_summary = read_inputs()

    writer = tf.summary.FileWriter('log')

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print(len(threads), 'threads started')
        try:
            # while not coord.should_stop():
            for i in range(1):
                image, label = sess.run(inputs)
                print(image.shape)
                print(label.shape)
                writer.add_summary(sess.run(image_summary))
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()
            coord.join(threads)
            writer.close()

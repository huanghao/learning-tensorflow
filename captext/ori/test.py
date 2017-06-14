# coding:utf-8

import numpy as np
from PIL import Image
import tensorflow as tf
from scipy.misc import imresize

import utils
from model import crack_captcha_cnn
from gen_captcha import gen_captcha_text_and_image
from const import *


def crack_captcha_single(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(CHECK_POINTS_DIR))

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * CHAR_SET_LEN + n] = 1
            i += 1

        return utils.vec2text(vector)


def crack_captcha(files=[]):
    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(CHECK_POINTS_DIR))
        predict = tf.argmax(
            tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        if not files:
            for i in range(100):
                text1, image = gen_captcha_text_and_image()
                if image.shape != (IMAGE_WIDTH, IMAGE_HEIGHT, 3):
                    print("error image size:", image.shape)
                    continue

                image = utils.convert2gray(image)
                image = image.flatten() / 255

                text_list = sess.run(predict,
                                     feed_dict={X: [image], keep_prob: 1})
                text = text_list[0].tolist()
                vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
                i = 0
                for n in text:
                    vector[i * CHAR_SET_LEN + n] = 1
                    i += 1

                predict_text = utils.vec2text(vector)
                print("正确: {}  预测: {} {}".format(text1, predict_text, text1 == predict_text))

        else:
            for i in files:
                im = np.array(Image.open(i))
                im = imresize(im, (IMAGE_WIDTH, IMAGE_HEIGHT))

                im = utils.convert2gray(im)
                im = im.flatten() / 255

                text_list = sess.run(predict, feed_dict={X: [im], keep_prob: 1})
                text = text_list[0].tolist()
                vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
                j = 0
                for n in text:
                    vector[j * CHAR_SET_LEN + n] = 1
                    j += 1

                predict_text = utils.vec2text(vector)
                print("filename: {}  predict: {}".format(i.split("/")[-1].split(".")[0], predict_text))


def gen_and_reco():
    while True:
        text, image = gen_captcha_text_and_image()
        if image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
            break

    image = utils.convert2gray(image)
    image = image.flatten() / 255
    predict_text = crack_captcha_single(image)

    print("正确: {}  预测: {}".format(text, predict_text))


def test_taobao():
    import glob
    files = glob.glob('data/taobao/*.jpg')
    crack_captcha(files)

if __name__ == "__main__":
    """
    import sys

    if len(sys.argv) < 2:
        gen_and_reco()
    else:
        cap = sys.argv[-1]
        text = cap.split("/")[-1].split(".")[0]

        im = np.array(Image.open(cap))
        print(">>>", im.shape)
        from scipy.misc import imresize
        im = imresize(im, (60, 160))
        print(">>>", im.shape)
        # im.save(text+"_.jpg", "JPEG")
        im = utils.convert2gray(im)
        im = im.flatten() / 255
        predict_text = crack_captcha_single(im)

        print("正确: {}  预测: {}".format(text, predict_text))

    """
    test_taobao()

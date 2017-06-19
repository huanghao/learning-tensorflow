import os
import time
import glob
import logging
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.python.training import session_run_hook

from vgg11 import Vgg11


NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = []
ALPHABET = []

MAX_CAPTCHA = 4
CHAR_SET_LEN = 10

IMAGE_WIDTH = 96
IMAGE_HEIGHT = 32
IMAGE_CHANNEL = 3

BATCH_SIZE = 64
LEARNING_RATE = 0.1
LOG_FREQUENCY = 5
SUMMARIES_DIR = "log"

files = glob.glob('test_data/images_simple/*')
filename_queue = tf.train.string_input_producer(files)
reader = tf.WholeFileReader()

def transform_label(s):
    _, ext = os.path.splitext(str(s))
    text = os.path.basename(str(s))
    text = text.replace(ext, "")
    text_len = len(text)

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1

    # print("label %s >>> %s" % (text, vector))
    return vector

def read_inputs():
    key, val = reader.read(filename_queue)

    image = tf.image.decode_png(val, IMAGE_CHANNEL)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_HEIGHT, IMAGE_WIDTH)

    label = tf.py_func(transform_label, [key], tf.float64)
    label = tf.reshape(label, [CHAR_SET_LEN*MAX_CAPTCHA])

    inputs = tf.train.shuffle_batch([image, label], batch_size=BATCH_SIZE,
                                    capacity=BATCH_SIZE*5, min_after_dequeue=BATCH_SIZE*2)

    return inputs

def make_batch():
    images, labels = tf.train.slice_input_producer([IMAGES, LABELS], shuffle=True)
    return tf.train.batch([images, labels], batch_size=BATCH_SIZE)

def compute_accuracy(y_conv, y_):
    results = []
    for i in range(MAX_CAPTCHA):
        i = i * CHAR_SET_LEN
        a = tf.slice(y_conv, [0, i], [-1, CHAR_SET_LEN])
        b = tf.slice(y_, [0, i], [-1, CHAR_SET_LEN])
        results.append(tf.equal(tf.argmax(a, 1), tf.argmax(b, 1)))

    return results

def main():
    x = tf.placeholder(tf.float32,
                       [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    y_ = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])

    _st = time.time()

    vgg = Vgg11()
    y_conv = vgg.build(x)

    global_step_tensor = tf.train.get_or_create_global_step()

    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv, labels=y_), name='loss')
    train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step_tensor)
    loss_op = tf.summary.scalar('loss', loss)

    correct_prediction = compute_accuracy(y_conv, y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    accuracy_op = tf.summary.scalar('accuracy', accuracy)

    summary_merged = tf.summary.merge_all()
    # batch_images, batch_labels = make_batch()
    inputs = read_inputs()

    class MyHook(session_run_hook.SessionRunHook):
        def before_run(self, run_context):
            # batch_x, batch_y = run_context.session.run([batch_images, batch_labels])
            batch_x, batch_y = run_context.session.run(inputs)
            return session_run_hook.SessionRunArgs(
                fetches=None, feed_dict={
                    x: batch_x,
                    y_: batch_y,
                })

    class LoggerHook(tf.train.SessionRunHook):
        def begin(self):
            self._step = -1
            self._start_time = time.time()

        def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs([loss, accuracy])  # Asks for loss value.

        def after_run(self, run_context, run_values):
            if self._step % LOG_FREQUENCY == 0:
                current_time = time.time()
                duration = current_time - self._start_time
                self._start_time = current_time

                loss_value, accuracy_value = run_values.results
                examples_per_sec = LOG_FREQUENCY * BATCH_SIZE / duration
                sec_per_batch = float(duration / LOG_FREQUENCY)

                format_str = ('%s: step %d, loss = %.2f, accuracy = %.2f'
                              '(%.1f examples/sec; %.3f sec/batch)')
                print (format_str % (datetime.now().time(), self._step, loss_value,
                                     accuracy_value, examples_per_sec, sec_per_batch))

    hooks = [
        LoggerHook(),
        tf.train.SummarySaverHook(5, output_dir=SUMMARIES_DIR, summary_op=summary_merged),
        MyHook(),
    ]

    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
        while not sess.should_stop():
            sess.run(train)

if __name__ == "__main__":
    main()


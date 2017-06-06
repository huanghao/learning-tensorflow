import os
import glob


import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = []
ALPHABET = []

MAX_CAPTCHA = 4
CHAR_SET_LEN = 10

IMAGE_WIDTH = 96
IMAGE_HEIGHT = 32
IMAGE_CHANNEL = 3

BATCH_SIZE = 2

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
    
    print("label %s >>> %s" % (text, vector))
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
    image_summary = tf.summary.image('image', inputs[0])

    return inputs, image_summary


inputs, image_summary = read_inputs()

writer = tf.summary.FileWriter('log')

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(len(threads), 'threads started')
    try:
        #while not coord.should_stop():
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


import glob

import tensorflow as tf
import numpy as np


from inference import train


def read_inputs():
    filename_queue = tf.train.string_input_producer(['data/train_data'])

    height, width = 28, 80
    label_bytes = 34 * 4
    image_bytes = height * width
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, val = reader.read(filename_queue)
    record_bytes = tf.decode_raw(val, tf.uint8)

    label = tf.reshape(
        tf.strided_slice(record_bytes, [0], [label_bytes]),
        [label_bytes, 1])

    image = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
        [height, width, 1])
    """
    image = tf.image.rgb_to_grayscale(image)
    height, width = 26, 78
    image = tf.random_crop(image, [height, width, 1])
    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.per_image_standardization(image)
    image.set_shape([height, width, 1])
    """

    sz = 1
    inputs = tf.train.shuffle_batch([image, label], batch_size=sz, capacity=sz*10, min_after_dequeue=sz*3)
    image_summary = tf.summary.image('image', inputs[0])
    return inputs, image_summary


inputs, image_summary = read_inputs()

writer = tf.summary.FileWriter('log')

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print len(threads), 'threads started'
    try:
        #while not coord.should_stop():
        for i in xrange(1000):
            image, label = sess.run(inputs)
            print image.shape, label.shape
            print label
            writer.add_summary(sess.run(image_summary))
    except tf.errors.OutOfRangeError:
        pass
    finally:
        coord.request_stop()
        coord.join(threads)
        writer.close()

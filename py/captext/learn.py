import glob

import tensorflow as tf
import numpy as np

def read_inputs():
    filename_queue = tf.train.string_input_producer(glob.glob('cap2/*.png'))
    reader = tf.WholeFileReader()
    key, val = reader.read(filename_queue)
    image = tf.reshape(
                tf.image.rgb_to_grayscale(
                    tf.image.decode_png(val)),
                [28, 80])
    inputs = tf.train.shuffle_batch([image, key], batch_size=2, capacity=100, min_after_dequeue=30)
    return inputs


inputs = read_inputs()

writer = tf.summary.FileWriter('log')

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print len(threads), 'threads started'
    try:
        #while not coord.should_stop():
        for i in range(2):
            image, label = sess.run(inputs)
            print image.shape, label.shape
            print label
    except tf.errors.OutOfRangeError:
        pass
    finally:
        coord.request_stop()
        coord.join(threads)
        writer.close()

# coding:utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from const import *
from model import crack_captcha_cnn
from inputs import get_next_batch
from inputs import read_inputs
import utils


def compute_loss(output):
    losses = []
    for i in range(MAX_CAPTCHA):
        i = i * CHAR_SET_LEN
        a = tf.slice(output, [0, i], [-1, CHAR_SET_LEN])
        b = tf.slice(Y, [0, i], [-1, CHAR_SET_LEN])
        l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=a, labels=b))
        losses.append(l)

    return tf.add_n(losses)


def train_crack_captcha_cnn():
    """训练"""
    output = crack_captcha_cnn()
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # loss = compute_loss(output)
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    tf.summary.scalar("loss", loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    inputs, _ = read_inputs()
    merged = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # saver.restore(sess, tf.train.latest_checkpoint(CHECK_POINTS_DIR))
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        step = 0
        try:
            while not coord.should_stop():
                batch_x, batch_y = sess.run(inputs)
                # batch_x, batch_y = get_next_batch(TRAIN_BATCH_SIZE)
                merged_op, _, loss_ = sess.run(
                    [merged, optimizer, loss],
                    feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})

                if step and step % 10 == 0:
                    utils.pprint(step, loss=loss_)
                    writer.add_summary(merged_op, step)

                # 每100 step计算一次准确率
                if step and step % 100 == 0:
                    batch_x_test, batch_y_test = sess.run(inputs)
                    # batch_x_test, batch_y_test = get_next_batch(TEST_BATCH_SIZE)
                    merged_op1, acc = sess.run(
                        [merged, accuracy],
                        feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                    writer.add_summary(merged_op1, step)
                    utils.pprint(step, accuracy=acc)
                    if acc > CHECK_POINTS_SAVE_ACCURACY:
                        saver.save(
                            sess, CHECK_POINTS_DIR + "crack_capcha_break.model", global_step=step)

                # 每1w步保存一次系数
                if step and step % CHECK_POINTS_SAVE_SEQ_STEPS == 0:
                    saver.save(
                        sess, CHECK_POINTS_DIR + "crack_capcha.model", global_step=step)

                step += 1

        except tf.errors.OutOfRangeError:
            print(">>>>>>>>>>>>>> OutOfRangeError <<<<<<<<<<<<<<")
        finally:
            coord.request_stop()
            coord.join(threads)
            writer.close()

if __name__ == '__main__':
    train_crack_captcha_cnn()

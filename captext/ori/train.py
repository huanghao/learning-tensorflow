# coding:utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import copy
from datetime import datetime

import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs

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
    global_step_tensor = tf.contrib.framework.get_or_create_global_step()
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # loss = compute_loss(output)
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # train 为了加快训练 learning_rate应该开始大，然后慢慢衰
    tf.summary.scalar("loss", loss)
    train = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(
        loss, global_step=global_step_tensor)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    inputs, _ = read_inputs(TRAIN_BATCH_SIZE)
    merged = tf.summary.merge_all()

    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(LOG_DIR + "/train")
    test_writer = tf.summary.FileWriter(LOG_DIR + "/test")

    class InputHook(session_run_hook.SessionRunHook):
        def before_run(self, run_context):
            batch_x, batch_y = get_next_batch(TRAIN_BATCH_SIZE)
            # batch_x, batch_y = run_context.session.run(inputs)
            return session_run_hook.SessionRunArgs(
                fetches=None, feed_dict={
                    X: batch_x,
                    Y: batch_y,
                    keep_prob: TRAIN_KEEP_DROP
                })

    class VerifyAccuracyHook(session_run_hook.SessionRunHook):
        def begin(self):
            self._start_time = time.time()
            l1 = [i for i in utils.frange(0.1, 0.6, 0.1)]
            l2 = [j for j in utils.frange(0.5, 1.1, 0.05)]
            self._acc_interval = l1 + l2
            self._global_step_tensor = training_util.get_global_step()

        def before_run(self, run_context):
            return SessionRunArgs(self._global_step_tensor)

        def after_run(self, run_context, run_values):
            global_step = run_values.results
            if not (global_step and
                    global_step % VERIFY_ACCURACY_STEPS == 0):
                return

            batch_x_test, batch_y_test = get_next_batch(TEST_BATCH_SIZE)
            merged_, acc = run_context.session.run(
                [merged, accuracy],
                feed_dict={X: batch_x_test, Y: batch_y_test,
                           keep_prob: TEST_KEEP_DROP})
            test_writer.add_summary(merged_, global_step)

            duration = time.time() - self._start_time
            info = utils.print_accuracy(global_step, duration, accuracy=acc)

            _tmp = copy.copy(self._acc_interval)
            self._acc_interval = [i for i in _tmp if i > acc]
            if self._acc_interval != _tmp:
                utils.logger.info(info)

            if acc >= CHECK_POINTS_SAVE_ACCURACY:
                saver.save(
                    run_context.session,
                    CHECK_POINTS_DIR + ACC_CHECKPOINT_BASENAME,
                    global_step=global_step)

                run_context.request_stop()  # 正确率达标，终止训练

    class LoggerHook(tf.train.SessionRunHook):
        def begin(self):
            self._start_time = time.time()
            self._global_step_tensor = training_util.get_global_step()

        def before_run(self, run_context):
            return tf.train.SessionRunArgs([self._global_step_tensor, loss])

        def after_run(self, run_context, run_values):
            global_step, loss_value = run_values.results
            if global_step % LOG_FREQUENCY == 0:
                current_time = time.time()
                duration = current_time - self._start_time
                self._start_time = current_time

                examples_per_sec = LOG_FREQUENCY * TRAIN_BATCH_SIZE / duration
                sec_per_batch = float(duration / LOG_FREQUENCY)

                format_str = ('%s step %d, loss = %f'
                              ' (%.1f examples/sec; %.3f sec/batch)')
                print (format_str % (datetime.now().strftime("%m-%d %H:%M:%S"),
                                     global_step, loss_value, examples_per_sec,
                                     sec_per_batch))

    hooks = [
        LoggerHook(),
        tf.train.CheckpointSaverHook(
            CHECK_POINTS_DIR, save_steps=CHECK_POINTS_SAVE_SEQ_STEPS,
            saver=saver, checkpoint_basename=CHECKPOINT_BASENAME),
        tf.train.SummarySaverHook(
            SUMMARY_SAVE_STEPS, summary_writer=train_writer, summary_op=merged
        ),
        VerifyAccuracyHook(),
        InputHook(),
    ]

    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
        while not sess.should_stop():
            sess.run(train)

if __name__ == '__main__':
    utils.print_const_info()
    train_crack_captcha_cnn()

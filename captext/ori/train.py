# coding:utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import string
import numpy as np
from PIL import Image
import tensorflow as tf

from datetime import datetime
from scipy.misc import imresize

from gen_captcha import gen_captcha_text_and_image

text, image = gen_captcha_text_and_image()
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = len(text)


def convert2gray(img):
    """把彩色图像转为灰度图像（色彩对识别验证码没有什么用）"""
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image【,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""

# 文本转向量
char_set = string.digits + string.ascii_letters + '_'  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set)
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    def char2pos(c):
        if c =='_':
            k = 62
            return k
        k = ord(c)-48
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
    return vector

# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i #c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx <36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx-  36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)

"""
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有
vec = text2vec("F5Sd")
text = vec2text(vec)
print(text)  # F5Sd
vec = text2vec("SFd5")
text = vec2text(vec)
print(text)  # SFd5
"""

# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i,:] = image.flatten() / 255 # (image.flatten()-128)/128  mean为0
        batch_y[i,:] = text2vec(text)

    return batch_x, batch_y

####################################################################

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32) # dropout

# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    #w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    #w_c2_alpha = np.sqrt(2.0/(3*3*32))
    #w_c3_alpha = np.sqrt(2.0/(3*3*64))
    #w_d1_alpha = np.sqrt(2.0/(8*32*64))
    #out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]))
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    #out = tf.nn.softmax(out)
    return out


def pprint(step, loss=None, accuracy=None):
    format_str = '%s: step %d, '
    if loss:
        format_str += 'loss = %.2f'
        print (format_str % (datetime.now(), step, loss))
    if accuracy:
        format_str += 'accuracy = %.2f'
        print (format_str % (datetime.now(), step, accuracy))

def compute_loss(output):
    losses = []
    for i in range(MAX_CAPTCHA):
        i = i * CHAR_SET_LEN
        a = tf.slice(output, [0, i], [-1, CHAR_SET_LEN])
        b = tf.slice(Y, [0, i], [-1, CHAR_SET_LEN])
        l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=a, labels=b))
        losses.append(l)

    return tf.add_n(losses)

# 训练
def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    # loss
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # loss = compute_loss(output)
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    tf.summary.scalar("loss", loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    merged = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./log", sess.graph)
        # test_writer = tf.summary.FileWriter("./log", sess.graph)

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            merged_op, _, loss_ = sess.run([merged, optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            if step % 10 == 0:
                pprint(step, loss=loss_)
                writer.add_summary(merged_op, step)
            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                merged_op1, acc = sess.run([merged, accuracy], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                writer.add_summary(merged_op1, step)
                pprint(step, accuracy=acc)
                # 如果准确率大于50%,保存模型,完成训练
                if acc > 0.98:
                    saver.save(sess, "ckpoints/crack_capcha_break.model", global_step=step)
            # 每1w步保存一次系数
            if step % 10000 == 0 and step:
                saver.save(sess, "ckpoints/crack_capcha.model", global_step=step)

            step += 1

def crack_captcha_single(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./ckpoints/'))

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i*CHAR_SET_LEN + n] = 1
            i += 1

        return vec2text(vector)

def crack_captcha(files=[]):
    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./ckpoints/'))
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        if not files:
            for i in range(100):
                text1, image = gen_captcha_text_and_image()
                if image.shape != (60, 160, 3):
                    print("error image size:", image.shape)
                    continue

                image = convert2gray(image)
                image = image.flatten() / 255

                text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
                text = text_list[0].tolist()
                vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
                i = 0
                for n in text:
                    vector[i*CHAR_SET_LEN + n] = 1
                    i += 1

                print("正确: {}  预测: {} {}".format(text1, vec2text(vector), text1 == vec2text(vector)))

        else:
            for i in files:
                im = np.array(Image.open(i))
                im = imresize(im, (60,160))

                im = convert2gray(im)
                im = im.flatten() / 255

                text_list = sess.run(predict, feed_dict={X: [im], keep_prob: 1})
                text = text_list[0].tolist()
                vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
                j = 0
                for n in text:
                    vector[j*CHAR_SET_LEN + n] = 1
                    j += 1

                print("filename: {}  predict: {}".format(i.split("/")[-1].split(".")[0], vec2text(vector)))


if __name__ == '__main__':
    train_crack_captcha_cnn()
    # text, image = gen_captcha_text_and_image()
    # image = convert2gray(image)
    # image = image.flatten() / 255
    # predict_text = crack_captcha(image)
    # print("正确: {}  预测: {}".format(text, predict_text))
    # crack_captcha()

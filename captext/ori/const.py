
import string
import tensorflow as tf

CHAR_SET = string.digits + string.ascii_letters
CHAR_SET1 = string.digits + string.ascii_letters + '_'

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160

MAX_CAPTCHA = 4
CHAR_SET_LEN = len(CHAR_SET1)

DEFAULT_BATCH_SIZE = 128
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 100

LEARNING_RATE = 0.001
LOG_DIR = "./log"
TRAIN_SETS = 'data/datasets/*'
CHECK_POINTS_DIR = LOG_DIR 
CHECK_POINTS_SAVE_SEQ_STEPS = 10000
CHECK_POINTS_SAVE_ACCURACY = 0.98
TRAIN_SET_NUM = 5000

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])

keep_prob = tf.placeholder(tf.float32)  # dropout

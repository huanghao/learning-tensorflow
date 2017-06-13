# coding:utf-8

import numpy as np
import string
import random

from PIL import Image
import matplotlib.pyplot as plt

from captcha.image import ImageCaptcha  # pip install captcha


def random_captcha_text(captcha_size=4):
    """验证码中的字符, 就不用汉字了;验证码一般都无视大小写；验证码长度4个字符"""

    return random.sample(string.digits + string.ascii_letters, captcha_size)


def gen_captcha_text_and_image(need_save=False):
    """生成字符对应的验证码"""
    image = ImageCaptcha()

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    if need_save:
        image.write(captcha_text, captcha_text + '.jpg')  # 写到文件

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)

    return captcha_text, captcha_image

if __name__ == '__main__':
    text, image = gen_captcha_text_and_image()
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)
    plt.show()

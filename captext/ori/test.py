import sys

import numpy as np
import tensorflow as tf 
from PIL import Image

from train import convert2gray
from train import crack_captcha_single, crack_captcha
from train import crack_captcha_cnn
from gen_captcha import gen_captcha_text_and_image

def gen_and_reco():
    text, image = gen_captcha_text_and_image()
    image = convert2gray(image)
    image = image.flatten() / 255
    predict_text = crack_captcha_single(image)
    print("正确: {}  预测: {}".format(text, predict_text))    


def test_taobao():
    import glob
    files = glob.glob('data/taobao/*')
    crack_captcha(files)
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        gen_and_reco()
    else:
        cap = sys.argv[-1]
        text = cap.split("/")[-1].split(".")[0]
        
        im = np.array(Image.open(cap))
        print(">>>", im.shape)
        from scipy.misc import imresize
        im = imresize(im, (60,160))
        print(">>>", im.shape)
        # im.save(text+"_.jpg", "JPEG")
        im = convert2gray(im)
        im = im.flatten() / 255
        predict_text = crack_captcha_single(im)
        
        print("正确: {}  预测: {}".format(text, predict_text))    

    """
    test_taobao()
    """
    

import sys
sys.path.insert(0, '/data1/huanghao/workspace/sources/tensorflow-vgg')

from flask import Flask, request, redirect, url_for, render_template
import tensorflow as tf
import numpy as np

import utils
import vgg16

synset = [l.strip() for l in open('/data1/huanghao/workspace/sources/tensorflow-vgg/synset.txt')]

test_image = tf.placeholder(tf.float32, [1, 224, 224, 3])

sess = tf.InteractiveSession()

vgg = vgg16.Vgg16()
vgg.build(test_image)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__)


def to_word(prob):
    prob = prob.flatten()
    pred = np.argsort(prob)[::-1][:5]
    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[p], prob[p], int(prob[p]*1000)) for p in pred]
    print(("Top5: ", top5))
    return top5


@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'GET':
        return '''
<!doctype html>
<title>Upload an image</title>
<h1>Upload an image</h1>
<form method=post enctype=multipart/form-data>
  <p><input type=file name=file />
     <input type=submit value=Upload />
</form>
'''

    if 'file' not in request.files or not request.files['file'].filename:
        return redirect(request.url)

    file = request.files['file']
    img = utils.load_image(file).reshape(-1, 224, 224, 3)
    
    prob = sess.run(vgg.prob, {test_image: img})
    return render_template('imagenet.html', pred=to_word(prob))
        


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=1, port=8080)

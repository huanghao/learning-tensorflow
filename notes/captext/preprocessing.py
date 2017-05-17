import struct
import glob

import cv2
import numpy as np


CHARS = {c: i for i, c in enumerate("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")}

def one_hot(idx, depth):
    v = np.zeros([len(idx), depth], dtype=np.uint8)
    for i, j in enumerate(idx):
        v[i, j] = 1
    return v


height, width = 28, 80
image_length = height * width
label_length = len(CHARS) * 4

with open('train_data', 'w') as f:
    for name in glob.glob('data/*.png')[:10]:
        image = cv2.imread(name, 0).flatten()

        text = name.split('/')[-1].split('.', 1)[0]
        label = one_hot([CHARS[c] for c in text], len(CHARS)).flatten()

        f.write(struct.pack('<' + str(label_length) + 'B', *label))
        f.write(struct.pack('<' + str(image_length) + 'B', *image))


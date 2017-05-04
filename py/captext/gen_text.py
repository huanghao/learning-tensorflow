import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


size = (80, 28)
chars = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
font = ImageFont.truetype('/Library/Fonts/Arial Black.ttf', 24)

for _ in xrange(1000):
    text = ''.join(random.sample(chars, 4))

    im = Image.new('RGB', size, (0,0,0))
    d = ImageDraw.Draw(im)
    d.text((0, 0), text, font=font, fill=(255,255,255))

    filename = text + '.png'
    print filename
    im.save(filename)

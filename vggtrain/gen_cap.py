import sys
import random

from PIL import Image, ImageDraw, ImageFont


if len(sys.argv) > 2:
    n = int(sys.argv)
else:
    n = 1

n = max(n, 1)
    
size = (96, 32)
chars = "0123456789"
IMAGE_DIR = "test_data/images_simple/"
font = ImageFont.truetype('ArialBlack.ttf', 24)

for _ in range(2):
    text = ''.join(random.sample(chars, 4))

    im = Image.new('RGB', size, (255,255,255))
    d = ImageDraw.Draw(im)
    d.text((0, 0), text, font=font, fill=(0,0,0))

    im.save(IMAGE_DIR + text + '.png')


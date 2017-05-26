import sys
import random

from PIL import Image, ImageDraw, ImageFont


if len(sys.argv) > 2:
    n = int(sys.argv)
else:
    n = 1
n = max(n, 1)
    
size = (80, 28)
chars = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
font = ImageFont.truetype('/Library/Fonts/Arial Black.ttf', 24)

for _ in xrange(n):
    text = ''.join(random.sample(chars, 4))

    im = Image.new('RGB', size, (0,0,0))
    d = ImageDraw.Draw(im)
    d.text((0, 0), text, font=font, fill=(255,255,255))

    filename = text + '.png'
    print filename
    im.save(filename)

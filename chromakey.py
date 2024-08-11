import os

import rawpy
from PIL import Image, ImageOps

path = 'data/RAW/SC1/BR1'
filename = os.path.join(path, '00570/back.orf')


raw = rawpy.imread(filename)
rgb_image = raw.postprocess()

image = Image.fromarray(rgb_image).convert('RGBA')

hsv_image = image.convert('HSV')
h, s, v = hsv_image.split()

green_mask = Image.eval(h, lambda x: 255 if 50 < x < 90 else 0)
green_mask = ImageOps.invert(green_mask)

rgba_image = image.copy()
rgba_image.putalpha(green_mask)
rgba_image.save('output.png', 'PNG')
import os

import rawpy
from PIL import Image, ImageOps

path = 'data/RAW/SC1/BR1'
filename = os.path.join(path, '00570/back.orf')

raw = rawpy.imread(filename).postprocess()
rgba = Image.fromarray(raw).convert('RGBA')

h, _, _ = rgba.convert('HSV').split()
mask = ImageOps.invert(Image.eval(h, lambda x: 255 if 50 < x < 90 else 0))

rgba.putalpha(mask)
rgba.save('output.png', 'PNG')
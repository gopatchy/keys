import os

from sklearn.cluster import KMeans
from wand.image import Image
import numpy as np


def chroma_key(img, object_colors=1, key_coords=(0,0)):
    hsv = _hsv_points(img)
    hsv_cart = _hsv_to_cartesian(hsv)
    labels = _cluster(hsv_cart, object_colors+1, img)
    mask = _mask(labels, key_coords)
    with Image.from_array(mask) as mask_img:
        img.composite_channel('all_channels', mask_img, 'multiply')


def _hsv_points(img):
    hsv_img = img.clone()
    hsv_img.transform_colorspace('hsv')
    pixels = list(hsv_img.export_pixels(channel_map='RGB'))  # this actually means HSV
    return np.array(pixels).reshape(-1, 3)


def _hsv_to_cartesian(hsv):
    h = hsv[:, 0]
    s = hsv[:, 1]
    v = hsv[:, 2]

    x = s * np.cos(h * 2 * np.pi / 255)
    y = s * np.sin(h * 2 * np.pi / 255)

    return np.column_stack((x, y, v))


def _cluster(points, num_clusters, img):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(points)
    return kmeans.labels_.reshape(img.height, img.width)


def _mask(labels, key_coords):
    key_label = labels[key_coords]
    return (labels != key_label).astype(np.uint8) * 255


path = 'data/RAW/SC1/BR1'
filename = os.path.join(path, '41244/back.orf')

with Image(filename=filename) as img:
    chroma_key(img)
    img.save(filename='masked_image.jpg')
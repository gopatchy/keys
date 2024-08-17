import os

from sklearn.cluster import KMeans
from wand.color import Color
from wand.image import Image
import numpy as np


def chroma_key(img, num_object_colors=1, key_box=((0,0),(100,100)), min_object_pixels=100):
    hsv = _hsv_points(img)
    hsv_cart = _hsv_to_cartesian(hsv)
    labels = _cluster(hsv_cart, num_object_colors+1, img)
    key_label = _choose_key(labels, key_box)
    mask = _create_mask(labels, key_label)
    ids = _islands(mask)
    _filter_islands(mask, ids, min_object_pixels)
    _apply_mask(img, mask)
    img.trim()


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


def _choose_key(labels, key_box):
    box_labels = labels[key_box[0][0]:key_box[1][0], key_box[0][1]:key_box[1][1]]
    values, counts = np.unique(box_labels.flatten(), return_counts=True)
    i = np.argmax(counts)
    return values[i]


def _islands(mask):
    ids = np.zeros_like(mask).astype(np.uint64)
    next_id = 1

    for (i, j), _ in np.ndenumerate(mask):
        _flood_fill(ids, mask, i, j, next_id)
        next_id += 1

    return ids


def _filter_islands(mask, ids, min_object_pixels):
    values, counts = np.unique(ids.flatten(), return_counts=True)
    ix = np.where(counts < min_object_pixels)
    mask[np.isin(ids, values[ix])] = 0


def _flood_fill(ids, mask, i, j, val):
    queue = [(i, j)]

    while queue:
        (i, j) = queue.pop()

        if i < 0 or i >= mask.shape[0] or j < 0 or j >= mask.shape[1]:
            continue

        if mask[i, j] == 0 or ids[i, j] != 0:
            continue

        ids[i, j] = val

        queue.append((i - 1, j))
        queue.append((i + 1, j))
        queue.append((i - 1, j - 1))
        queue.append((i + 1, j - 1))
        queue.append((i - 1, j + 1))
        queue.append((i + 1, j + 1))
        queue.append((i, j - 1))
        queue.append((i, j + 1))


def _create_mask(labels, key_label):
    return (labels != key_label).astype(np.uint8) * 255


def _apply_mask(img, mask):
    with Image.from_array(mask) as mask_img:
        img.alpha_channel = 'activate'
        img.composite_channel('alpha', mask_img, 'copy_alpha', 0, 0)
        img.background_color = Color('transparent')


path = 'data/RAW/SC1/BR1'
filename = os.path.join(path, '41244/back.orf')

with Image(filename=filename) as img:
    img.crop(left=0, top=0, width=4000, height=3000)
    chroma_key(img)
    img.save(filename='chromakey.png')
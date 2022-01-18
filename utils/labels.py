import pandas as pd
import numpy as np
import os


classes = pd.read_csv("dataset/classes.csv", index_col='name')


# [

#     ((1,2,3),(2,3,4)),

# ]


mapping_20 = {

    (111, 74, 0): (0, 0, 0),
    (81, 0, 81): (0, 0, 0),
    (250, 170, 160): (0, 0, 0),
    (230, 150, 140): (0, 0, 0),
    (180, 165, 180): (0, 0, 0),
    (150, 100, 100): (0, 0, 0),
    (150, 120, 90): (0, 0, 0),
    (153, 153, 153): (0, 0, 0),
    (0,  0, 90): (0, 0, 0),
    (0,  0, 110): (0, 0, 0),
    (0, 0, 142): (0, 0, 0),
    (128, 64, 128): (128, 64, 128),
    (244, 35, 232): (244, 35, 232),
    (70, 70, 70): (70, 70, 70),
    (102, 102, 156): (102, 102, 156),
    (190, 153, 153): (190, 153, 153),
    (153, 153, 153): (153, 153, 153),
    (250, 170, 30): (250, 170, 30),
    (220, 220,  0): (220, 220,  0),
    (107, 142, 35): (107, 142, 35),
    (152, 251, 152): (152, 251, 152),
    (70, 130, 180): (70, 130, 180),
    (220, 20, 60): (220, 20, 60),
    (255,  0,  0): (255,  0,  0),
    (0,  0, 142): (0,  0, 142),
    (0,  0, 70): (0,  0, 70),
    (0, 60, 100): (0, 60, 100),
    (0, 80, 100): (0, 80, 100),
    (0,  0, 230): (0,  0, 230),
    (119, 11, 32): (119, 11, 32),

}


def encode_labels(mask):
    label_mask = np.zeros_like(mask)

    for k in mapping_20:

        # print(mask[mask == k].shape)
        label_mask[mask == k] = mapping_20[k]
    return mask


cls2rgb = {cl: list(classes.loc[cl, :]) for cl in classes.index}
idx2rgb = {idx: np.array(rgb)
           for idx, (cl, rgb) in enumerate(cls2rgb.items())}


def map_class_to_rgb(p):
    return idx2rgb[p[0]]


def adjust_mask(mask, flat=False):

    semantic_map = []
    for colour in list(cls2rgb.values()):
        equality = np.equal(mask, colour)  # 256x256x3 with True or False
        # 256x256 If all True, then True, else False
        class_map = np.all(equality, axis=-1)
        # List of 256x256 arrays, map of True for a given found color at the pixel, and False otherwise.
        semantic_map.append(class_map)
    # 256x256x32 True only at the found color, and all False otherwise.
    semantic_map = np.stack(semantic_map, axis=-1)
    if flat:
        semantic_map = np.reshape(semantic_map, (-1, 128*128))

    return np.float16(semantic_map)  # convert to numbers

# -*- coding: utf-8 -*-

from utils.colourconv import rgb2grey, rgb2hed
from skimage import img_as_ubyte

def image_colour_convert(img, color_mode='rgb'):
    if color_mode == 'rgb':
        return img
    elif color_mode == 'multispectral':
        return img
    elif color_mode == 'greyscale':
        return rgb2grey(img / 255)[..., None]
    elif color_mode == 'haemotoxylin':
        return rgb2hed(img / 255)[..., [0]]
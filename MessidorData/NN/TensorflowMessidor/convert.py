"""Resize and crop images to square, save as tiff."""
"""https://github.com/sveitser/kaggle_diabetic/blob/master/convert.py"""
import os
from multiprocessing.pool import Pool

import numpy as np
#from PIL import Image, ImageFilter
import PIL.Image as Image
import PIL.ImageFilter as ImageFilter
import settings
import cv2

N_PROC = 2

def convert(fname, crop_size):
    img = Image.open(fname)

    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    h, w, _ = ba.shape

    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)

        foreground = (ba > max_bg + 10).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()

        if bbox is None:
            print('bbox none for {} (???)'.format(fname))
        else:
            left, upper, right, lower = bbox
            # if we selected less than 80% of the original
            # height, just crop the square
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print('bbox too small for {}'.format(fname))
                bbox = None
    else:
        bbox = None

    if bbox is None:
        bbox = square_bbox(img)

    cropped = img.crop(bbox)
    resized = cropped.resize([crop_size, crop_size])
    return resized


def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


def convert_square(fname, crop_size):
    img = Image.open(fname)
    bbox = square_bbox(img)
    cropped = img.crop(bbox)
    resized = cropped.resize([crop_size, crop_size])
    return resized


def get_convert_fname(fname, extension, directory, convert_directory):
    return fname.replace('jpeg', extension).replace(directory,
                                                    convert_directory)


def process(args):
    fun, arg = args
    directory, convert_directory, fname, crop_size, extension = arg
    convert_fname = get_convert_fname(fname, extension, directory,
                                      convert_directory)
    if not os.path.exists(convert_fname):
        img = fun(fname, crop_size)
        save(img, convert_fname)


def save(img, fname):
    img.save(fname, quality=97)


def main(directory, convert_directory, crop_size, extension):

    try:
        os.mkdir(convert_directory)
    except OSError:

        pass

    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(directory)
                 for f in fn if f.endswith('jpg') or f.endswith('tif') or f.endswith('png')]
    filenames = sorted(filenames)

    for f in filenames:
        img = convert(f, crop_size)
        #b = np.zeros(np.asarray(img).shape)
        #cv2.circle(b, (img.size[1] / 2, img.size[0] / 2), int(512 * 0.9), (1, 1, 1), -1, 8, 0)
        #im_blur = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 512 / 30), -4, 128) * b + 128 * (1 - b)
        save(img, os.path.join(convert_directory, f))




if __name__ == '__main__':
    # main(settings.dataDirectoryPath, settings.convertDataDirectoryPath, crop_size=512, extension='tif')
    main("/home/devrim/Downloads/normal", "/home/devrim/Downloads/normal", crop_size=512, extension='png')

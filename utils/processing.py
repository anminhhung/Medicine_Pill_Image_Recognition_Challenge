# Set up custom environment before nearly anything else is imported

import os
import copy
import albumentations as A
import cv2, random


def pad_images_to_same_size(images):
        h, w = images.shape[:2]
        width_max  = max(h , w)
        height_max = max(w, h)
        diff_vert = height_max - h
        pad_top = diff_vert//2
        pad_bottom = diff_vert - pad_top
        diff_hori = width_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left
        img_padded = cv2.copyMakeBorder(images, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        return img_padded


def pad_plus_resize(img, size):
    img = pad_images_to_same_size(img)
    img = cv2.resize(img, (size, size))
    return img




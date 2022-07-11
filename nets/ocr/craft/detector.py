import numpy as np
import cv2
import torch
import os
import sys

from logging import getLogger
from PIL import Image

from .detection import get_detector, get_textbox
from .utils import group_text_box, diff

LOGGER = getLogger(__name__)

class CRAFT_DETECTOR(object):
    def __init__(self, detector_path, gpu='cpu', verbose=True, quantize=True):
        if gpu == 'cpu':
            self.device = 'cpu'
            LOGGER.warning('Using CPU. Note: This module is much faster with a GPU.')
        elif gpu == 'cuda':
            self.device = 'cuda'
        elif not torch.cuda.is_available():
            self.device = 'cpu'
            LOGGER.warning('CUDA not available - defaulting to CPU. Note: This module is much faster with a GPU.')
        else:
            self.device = gpu

        self.detector = get_detector(detector_path, self.device, quantize)

    def detect(self, img, min_size = 20, text_threshold = 0.7, low_text = 0.4,\
               link_threshold = 0.4,canvas_size = 2560, mag_ratio = 1.,\
               slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
               width_ths = 0.5, add_margin = 0.1, optimal_num_chars=None):

        text_box = get_textbox(self.detector, img, canvas_size, mag_ratio,\
                               text_threshold, link_threshold, low_text,\
                               False, self.device, optimal_num_chars)

        horizontal_list, free_list = group_text_box(text_box, slope_ths,\
                                                    ycenter_ths, height_ths,\
                                                    width_ths, add_margin, \
                                                    (optimal_num_chars is None))

        if min_size:
            horizontal_list = [i for i in horizontal_list if max(i[1]-i[0],i[3]-i[2]) > min_size]
            free_list = [i for i in free_list if max(diff([c[0] for c in i]), diff([c[1] for c in i]))>min_size]

        return horizontal_list, free_list

    def readtext(self, image, min_size = 20, text_threshold = 0.7, low_text = 0.4,\
                 link_threshold = 0.4, canvas_size = 2560, mag_ratio = 1.,\
                 slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                 width_ths = 0.5, add_margin = 0.1):

        horizontal_list, free_list = self.detect(image, min_size, text_threshold,\
                                                 low_text, link_threshold,\
                                                 canvas_size, mag_ratio,\
                                                 slope_ths, ycenter_ths,\
                                                 height_ths,width_ths,\
                                                 add_margin, False)

        return horizontal_list, free_list
    
    '''
        get cv2 image -> crop -> convert PIL image.
    '''
    def crop_image(self, image, bbox):
        height, width, _ = image.shape
        x_min = max(0, bbox[0])
        x_max = min(bbox[1], width)
        y_min = max(0, bbox[2])
        y_max = min(bbox[3], height)

        crop_image = image[y_min:y_max, x_min:x_max]

        # convert pil
        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(crop_image)

        return im_pil
    
    def visualize_box_text(self, image, bbox, text, color_box=(0, 255, 255), color_text=(0, 0, 255)):
        height, width, _ = image.shape
        x_min = max(0, bbox[0])
        x_max = min(bbox[1], width)
        y_min = max(0, bbox[2])
        y_max = min(bbox[3], height)

        bbox = [x_min, y_min, x_max, y_max]
        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_box, 1)

        image = cv2.putText(image, text, (bbox[0], bbox[1]), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1, cv2.LINE_AA)

        return image
    
    def draw_bbox(self, image, bbox, color_box=(0, 255, 255)):
        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_box, 1)

        return image

    def visualize_bbox(self, image, horizontal_list, color=(0, 255, 255), visualize_text=False):
        height, width, _ = image.shape
        for bbox in horizontal_list:
            x_min = max(0, bbox[0])
            x_max = min(bbox[1], width)
            y_min = max(0, bbox[2])
            y_max = min(bbox[3], height)

            image = self.draw_bbox(image, [x_min, y_min, x_max, y_max], color)
            if visualize_text:
                image = cv2.putText(image, '{} {} {} {}'.format(x_min, y_min, x_max, y_max), (x_min, y_min), \
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        return image
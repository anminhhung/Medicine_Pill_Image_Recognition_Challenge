# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import time
import sys
from PIL import Image, ImageDraw, ImageFont
from config import  *

# text detection
import prescription.text_detector.ppocr.utility as utility
from prescription.text_detector.ppocr.utils.logging import get_logger
from prescription.text_detector.ppocr.utils.utility import get_image_file_list, check_and_read_gif
from prescription.text_detector.ppocr.data import create_operators, transform
from prescription.text_detector.ppocr.postprocess import build_post_process
logger = get_logger()

# text recognition
from prescription.text_recognizer.vietocr.vietocr.tool.predictor import Predictor
from prescription.text_recognizer.vietocr.vietocr.tool.config import Cfg
# config = Cfg.load_config_from_file(BASE_DIR+ '/prescription/text_recognizer/vietocr/config/base.yml')
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = BASE_DIR + "/prescription/text_recognizer/models/transformerocr.pth" #"1sWoYLzyk3MalZbNyBCf8WXsyTCEfBoxh" #
config['cnn']['pretrained'] = False
config['device'] = 'cuda:0' #'cuda:0' #
config['predictor']['beamsearch']= False
recognizer = Predictor(config)

# Key information extraction
import math
import difflib
import pandas as pd
import csv
mapping_file = BASE_DIR + "/prescription/label-name.csv"
font_text = ImageFont.truetype(BASE_DIR+ "/prescription/font/SVN-Arial-Regular.ttf", 20)


class TextDetector(object):
    def __init__(self, args):
        self.args = args
        self.det_algorithm = args.det_algorithm
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': args.det_limit_side_len,
                'limit_type': args.det_limit_type
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {}
        if self.det_algorithm == "DB":
            postprocess_params['name'] = 'DBPostProcess'
            postprocess_params["thresh"] = args.det_db_thresh
            postprocess_params["box_thresh"] = args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
            postprocess_params["use_dilation"] = True
        elif self.det_algorithm == "EAST":
            postprocess_params['name'] = 'EASTPostProcess'
            postprocess_params["score_thresh"] = args.det_east_score_thresh
            postprocess_params["cover_thresh"] = args.det_east_cover_thresh
            postprocess_params["nms_thresh"] = args.det_east_nms_thresh
        elif self.det_algorithm == "SAST":
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'resize_long': args.det_limit_side_len
                }
            }
            postprocess_params['name'] = 'SASTPostProcess'
            postprocess_params["score_thresh"] = args.det_sast_score_thresh
            postprocess_params["nms_thresh"] = args.det_sast_nms_thresh
            self.det_sast_polygon = args.det_sast_polygon
            if self.det_sast_polygon:
                postprocess_params["sample_pts_num"] = 6
                postprocess_params["expand_scale"] = 1.2
                postprocess_params["shrink_ratio_of_width"] = 0.2
            else:
                postprocess_params["sample_pts_num"] = 2
                postprocess_params["expand_scale"] = 1.0
                postprocess_params["shrink_ratio_of_width"] = 0.3
        else:
            logger.info("unknown det_algorithm:{}".format(self.det_algorithm))
            sys.exit(0)

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors = utility.create_predictor(
            args, 'det', logger)  # paddle.jit.load(args.det_model_dir)
        # self.predictor.eval()

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points


    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        starttime = time.time()

        self.input_tensor.copy_from_cpu(img)
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)

        preds = {}
        if self.det_algorithm == "EAST":
            preds['f_geo'] = outputs[0]
            preds['f_score'] = outputs[1]
        elif self.det_algorithm == 'SAST':
            preds['f_border'] = outputs[0]
            preds['f_score'] = outputs[1]
            preds['f_tco'] = outputs[2]
            preds['f_tvo'] = outputs[3]
        elif self.det_algorithm == 'DB':
            preds['maps'] = outputs[0]
        else:
            raise NotImplementedError

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        if self.det_algorithm == "SAST" and self.det_sast_polygon:
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        elapse = time.time() - starttime
        return dt_boxes, elapse

def crop_bbox(img, box):
    width = int(box[1][0] - box[0][0])
    height = int(box[3][1] - box[0][1])
    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    top_left = [0, 0]
    top_right = [width - 1, 0]
    bottom_right = [width - 1, height - 1]
    bottom_left = [0, height - 1]

    dst_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    # Get the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped

def write_text(image, result_text, point):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw =  ImageDraw.Draw(pil_img)
    point = (point[0], int(point[1]-20))
    draw.text(point, result_text, (255, 0, 0), font=font_text)
    cv2_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return cv2_img

def get_all_drugnames(mapping_file):
    df = pd.read_csv(mapping_file)
    DrugList = df['Drug name'].tolist()
    # LabelCells = df['label'].tolist()
    # print(LabelCells)
    AllDrugNames = []
    for cell in DrugList:
        try:
            names = cell.split(" || ")
            for name in names:
                name = name.replace('"', '')
                # print("'"+name+"'")
                AllDrugNames.append(name)
        except:
            if math.isnan(cell):
                print(cell)
    return DrugList, AllDrugNames

if __name__ == "__main__":
    args = utility.parse_args()
    from prescription.config import det_model_dir, raw_img_dir, det_visualize, \
        det_out_viz_dir, det_db_thresh, det_db_box_thresh

    args.image_dir = raw_img_dir
    args.det_model_dir = det_model_dir
    args.det_db_thresh = det_db_thresh
    args.det_db_box_thresh = det_db_box_thresh
    args.use_gpu = True

    # print(args)
    image_file_list = get_image_file_list(args.image_dir)
    text_detector = TextDetector(args)

    DrugList, AllDrugNames = get_all_drugnames(mapping_file)
    # print(DrugList)

    header = ['PresName', 'DrugName', 'labels']
    PresOutput = open('public_test_pres_ouput_predict.csv', 'w', encoding='UTF-8', newline='')
    writer = csv.writer(PresOutput)
    writer.writerow(header)

    for image_file in image_file_list:
        print(40*"*")
        t1 = time.time()
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        src_im = img.copy()
        presname = os.path.basename(image_file)
        dt_boxes, elapse = text_detector(img)
        # print("boxes: ", dt_boxes[0], elapse)

        img_name_pure = os.path.split(image_file)[-1]

        drugname_boxes = []
        for box in dt_boxes:
            warp = crop_bbox(img, box)
            # text = recognizer.predict(warp)
            img_pil = Image.fromarray(warp)
            text = recognizer.predict(img_pil)
            # print("OCR Predict: ", text)

            # Key information extraction by rule
            drugnames_predict = difflib.get_close_matches(text, AllDrugNames)
            probs = []
            for name in drugnames_predict:
                prob = difflib.SequenceMatcher(None, text, name).ratio()
                probs.append(prob)
            if drugnames_predict != []:
                drugname = drugnames_predict[np.argmax(probs)]
                # print("drugnames_predict: ", drugnames_predict)
                drugnameShow = drugname + " - ID: "

                labels = ""
                ids = []
                for index, drug in enumerate(DrugList):
                    if drugname in drug:
                        ids.append(index)
                maxid = len(ids) - 1
                for i, id in enumerate(ids):
                    if i < maxid:
                        drugnameShow += f'{id},'
                        labels += f'{id}-'
                    else:
                        drugnameShow += f'{id}'
                        labels += f'{id}'
                writer.writerow([presname, drugname, labels])

                drugname_boxes.append(box)
                src_im = write_text(src_im, drugnameShow, (int(box[0][0]), int(box[0][1])))
        src_im = utility.draw_text_det_img(drugname_boxes, src_im)

        if det_visualize:
            img_path = os.path.join(det_out_viz_dir, "{}".format(img_name_pure))
            cv2.imwrite(img_path, src_im)
            logger.info("The visualized image saved in {}".format(img_path))

        infer_time = time.time() - t1
        logger.info("Predict time of {}: {}".format(presname, infer_time))


        cv2.imshow("pres", src_im)
        if cv2.waitKey(0) == ord("q"):
            break
    cv2.destroyAllWindows()

    PresOutput.close()

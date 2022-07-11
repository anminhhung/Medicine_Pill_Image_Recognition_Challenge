import cv2
import os 
import glob2
from tqdm import tqdm

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from nets import CRAFT_DETECTOR
from configs.config import init_config

# config
CFG = init_config()
DEVICE = CFG["device"]["device"]
craft_weight_path = CFG["craft"]["weight_path"]
vietocr_weight_path = CFG["vietocr"]["weight_path"]

# TEXT_DETECTOR
TEXT_DETECTOR = CRAFT_DETECTOR(craft_weight_path, DEVICE)

# TEXT RECOGNITION
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = vietocr_weight_path
config['cnn']['pretrained']=False
config['device'] = CFG["device"]["device"]
config['predictor']['beamsearch']=False

TEXT_RECOGNIZER= Predictor(config)

def predict_image(image_path, visual=False, result_dir="saved/ocr_results"):
    image_name = (image_path.split("/")[-1]).split(".")[0]
    image = cv2.imread(image_path)
    if visual:
        image_visual = image.copy()

    # detect 
    horizontal_list, free_list = TEXT_DETECTOR.readtext(image)

    list_bbox = []
    list_text = []

    for bbox in horizontal_list:
        crop_pil_image = TEXT_DETECTOR.crop_image(image, bbox)

        result_text = TEXT_RECOGNIZER.predict(crop_pil_image)

        list_bbox.append(bbox)
        list_text.append(result_text)

        # visualize
        if visual:
            image_visual = TEXT_DETECTOR.visualize_box_text(image_visual, bbox, result_text)

    if visual:
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        visual_image_path = os.path.join(result_dir, image_name + ".jpg")
        cv2.imwrite(visual_image_path, image_visual)

    return list_bbox, list_text

if __name__ == "__main__":
    list_bbox, list_text = predict_image("data/OCR_imprint/demo.jpg")
    print(list_text)
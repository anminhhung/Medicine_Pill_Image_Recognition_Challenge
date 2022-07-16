import sys
sys.path.append('..')

import cv2
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from nets import Segmentation_Detectron2
from configs.config import init_config

if __name__ == "__main__":
    print("[INFO] Starting........")
    segmentation = Segmentation_Detectron2('/nets/segmentation/dataset/label.json','/nets/segmentation/dataset/train_images','pill')
    segmentation.visualize_prediction(['/nets/segmentation/dataset/test/2.jpg'], segmentation.pill_metadata)

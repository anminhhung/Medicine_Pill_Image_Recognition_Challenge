import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.utils.visualizer import ColorMode
# import some common libraries
import numpy as np
import os, json, cv2, random, math,glob
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from .detectron2 import model_zoo
from .detectron2.data import MetadataCatalog, DatasetCatalog
from .detectron2.data.datasets import register_coco_instances
from .detectron2.utils.visualizer import Visualizer
from .detectron2.engine import DefaultTrainer
from .detectron2.engine import DefaultPredictor
from .detectron2.config import get_cfg


class Segmentation_Detectron2(object):
      
    def __init__(self, json_path,images_path,name_data):
        self.model="configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

        self.weight='model_final.pth'

        self.name_dataset='pill'

        self.lr=0.0001

        self.max_iter=1000

        self.bs=32

        self.cfg = get_cfg()

        self.cfg.merge_from_file(self.model)

        self.cfg.DATASETS.TRAIN = (self.name_dataset,)    

        self.cfg.DATALOADER.NUM_WORKERS = 2
        
        self.cfg.MODEL.WEIGHTS = self.weight #"detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"  # initialize from model zoo 
        
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        
        self.cfg.SOLVER.BASE_LR = self.lr
        
        self.cfg.SOLVER.MAX_ITER = (self.max_iter)

        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (self.bs)  # faster, and good enough for this toy dataset

        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes (pill)

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95   # set the testing threshold for this model

        self.pill_metadata = self.regist_data(json_path,images_path,name_data)

    # Hàm đăng kí data, nhận đầu vào là đường dẫn file json ( file gán nhãn ), đường dẫn file chứa các ảnh, tên đăng kí ( em thường để là 'pill' )
    def regist_data(self,json_path,images_path,name_data):  
      register_coco_instances(name_data, {}, json_path, images_path)
      dataset_dicts = DatasetCatalog.get(name_data)
      metadata = MetadataCatalog.get(name_data)
      return metadata

    # Hàm visualize dự đoán đầu ra của 1 tập ảnh, đầu vào là list đường dẫn ảnh, database là đầu ra của hàm regist_data bên trên ( data được đăng kí )
    def visualize_prediction(self,images_path, metadata):
        os.makedirs('output_images',exist_ok=True)
        for image_path in images_path:
          im = cv2.imread(image_path)
          outputs = self.instance_segment(image_path)
          v = Visualizer(im[:, :, ::-1],
                    metadata = metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                      )
          v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
          #cv2_imshow((v.get_image()[:, :, ::-1]))
          image_path = image_path.split('/')[-1]
          image_path = image_path.split('.')[0]
          cv2.imwrite('output_images/'+image_path+'output'+'.jpg',v.get_image()[:,:,::-1])

    # Hàm train, nhận đầu vào mà file model.yaml, file weight ( của em pretrained, mọi người có thể dùng file riêng ), tên data 'pill' ( cái này phải trùng với tên được đăng kí)
    def train(self):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    # Hàm xử lý ( được sài bên trong các hàm khác, không sài trực tiếp )
    def instance_segment(self,img_path):  # đầu vào là ảnh RGB, đầu ra là instance
        predictor = DefaultPredictor(self.cfg)

        im = cv2.imread(img_path)
        outputs = predictor(im)
        return outputs

    # Hàm nhận đầu vào là list các đường dẫn ảnh, trả ra list từng viên thuốc trong các ảnh đó
    def segmented_pills(self,image_paths,resize = True, size = (60,60)): 
        output_final = []
        for img_path in image_paths: 
            predict = self.instance_segment(img_path) # Dự đoán đầu ra ( sau khi instance segmentation )
            predict = predict["instances"].to("cpu")

            output = []
            for i in range(len(predict)): # Lấy từng pill segment được 
                x1,y1,x2,y2 = predict[i].pred_boxes.tensor[0]   # Lấy ra bounding box
                x1,y1,x2,y2 = math.floor(x1),math.floor(y1),math.floor(x2),math.floor(y2) # Chuyển về số nguyên
                
                moi = predict[i].pred_masks.squeeze()[y1:y2,x1:x2]

                im = cv2.imread(img_path)
                con = im[y1:y2,x1:x2]
                con = con*np.array(moi.unsqueeze(2))
                if resize:
                  con = cv2.resize(con,size)
                output_final.append(con)
            #  output_final.append(output)
        return output_final

# Hướng dẫn sử dụng :

# Muốn visualize prediction của 1 tập ảnh ( tạo thư mục chứa các ảnh được predict ), làm 2 bước :
# Bước 1 : Đăng kí data, database = regist_data(json_path, images_path, name_data)        
# Bước 2 : Dùng hàm visualize_prediction(images_path,database)

# Muốn trả ra list chứa từng viên được instance segment :
# Dùng hàm tra_ra_tung_vien_thuoc(input_image, resize = True, size = (28,28))     

# Muốn train :
# Bước 1 : Đăng kí data, database = regist_data(json_path, images_path, name_data)  ( Nếu đăng kí rồi thì không cần nữa ạ )   
# Bước 2 : Dùng hàm train(), sau khi train xong, có được một file weight, có thể dùng để đi predict hoặc ko có thể dùng file weight có sẵn của em ( Vinh )

if __name__ == '__main__':
    segmentation = Segmentation_Detectron2('dataset/label.json','dataset/train_images','pill')
    segmentation.visualize_prediction(['dataset/test/2.jpg'], segmentation.pill_metadata)
    # pills = tra_ra_tung_vien_thuoc(['dataset/test/2.jpg'])
    # count = 0 
    # for pill in pills:
    #     cv2.imwrite(f'output_images/{count}.jpg',pill)
    #     count += 1
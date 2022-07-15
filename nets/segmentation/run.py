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
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


# Hàm đăng kí data, nhận đầu vào là đường dẫn file json ( file gán nhãn ), đường dẫn file chứa các ảnh, tên đăng kí ( em thường để là 'pill' )
def regist_data(json_path,images_path,name_data):  
  register_coco_instances(name_data, {}, json_path, images_path)
  dataset_dicts = DatasetCatalog.get(name_data)
  pill_metadata = MetadataCatalog.get(name_data)
  return pill_metadata

# Hàm visualize dự đoán đầu ra của 1 tập ảnh, đầu vào là list đường dẫn ảnh, database là đầu ra của hàm regist_data bên trên ( data được đăng kí )
def visualize_prediction(images_path, database):
    os.makedirs('output_images',exist_ok=True)
    for image_path in images_path:
      im = cv2.imread(image_path)
      outputs = instance_segment(image_path)
      v = Visualizer(im[:, :, ::-1],
                metadata = database, 
                scale=1., 
                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                  )
      v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
      #cv2_imshow((v.get_image()[:, :, ::-1]))
      image_path = image_path.split('/')[-1]
      image_path = image_path.split('.')[0]
      cv2.imwrite('output_images/'+image_path+'output'+'.jpg',v.get_image()[:,:,::-1])
    return 0

# Hàm train, nhận đầu vào mà file model.yaml, file weight ( của em pretrained, mọi người có thể dùng file riêng ), tên data 'pill' ( cái này phải trùng với tên được đăng kí)
def train(model="configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",weight='model_final123.pth',dataset='pill',lr=0.0001,max_iter=1000,bs=32):         
     cfg = get_cfg()

     cfg.merge_from_file(model)

     cfg.DATASETS.TRAIN = (dataset,)    

     cfg.DATALOADER.NUM_WORKERS = 2
     
     cfg.MODEL.WEIGHTS = weight #"detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"  # initialize from model zoo 
    
     cfg.SOLVER.IMS_PER_BATCH = 2
    
     cfg.SOLVER.BASE_LR = lr
    
     cfg.SOLVER.MAX_ITER = (max_iter)

     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (bs)  # faster, and good enough for this toy dataset

     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes (pill)

     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
     trainer = DefaultTrainer(cfg)
     trainer.resume_or_load(resume=False)
     trainer.train()

# Hàm xử lý ( được sài bên trong các hàm khác, không sài trực tiếp )
def instance_segment(input,model="configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",weight='model_final123.pth'):  # đầu vào là ảnh RGB, đầu ra là instance
     cfg = get_cfg()

     cfg.merge_from_file(model)
     
     cfg.MODEL.WEIGHTS = weight # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    
     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95   # set the testing threshold for this model
     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

     predictor = DefaultPredictor(cfg)

     im = cv2.imread(input)
     outputs = predictor(im)
     return outputs

# Hàm nhận đầu vào là list các đường dẫn ảnh, trả ra list từng viên thuốc trong các ảnh đó
def tra_ra_tung_vien_thuoc(input_image,resize = True, size = (28,28)): 
    output_final = []
    for image in input_image: 
         predict = instance_segment(image) # Dự đoán đầu ra ( sau khi instance segmentation )
         predict = predict["instances"].to("cpu")

         output = []
         for i in range(len(predict)): # Lấy từng pill segment được 
             x1,y1,x2,y2 = predict[i].pred_boxes.tensor[0]   # Lấy ra bounding box
             x1,y1,x2,y2 = math.floor(x1),math.floor(y1),math.floor(x2),math.floor(y2) # Chuyển về số nguyên
             
             moi = predict[i].pred_masks.squeeze()[y1:y2,x1:x2]

             im = cv2.imread(image)
             con = im[y1:y2,x1:x2]
             con = con*np.array(moi.unsqueeze(2))
             if resize:
               con = cv2.resize(con,size)
             output.append(con)
         output_final.append(output)
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
    database = regist_data('dataset/label.json','dataset/train_images','pill')
    visualize_prediction(['dataset/test/2.jpg'],database)
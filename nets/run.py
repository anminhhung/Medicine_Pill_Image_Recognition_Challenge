from __future__ import print_function, division

import torch
import torch.nn as nn
from efficientnet.model import EfficientNet
from PIL import Image
import numpy as np
from sklearn.utils import compute_class_weight
from torchvision import datasets, models, transforms
import math
from segmentation.segmentation_detector import Segmentation_Detectron2
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    segmentation = Segmentation_Detectron2('./segmentation/dataset/label.json','./segmentation/dataset/train_images','pill')
    model = EfficientNet.from_pretrained('efficientnet-b7')
    class_num = 108
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, class_num)
    # model = EfficientNet.from_pretrained('efficientnet-b7').to(device)
    state = torch.load('./pill_ckp.pt')
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)
    model.eval()

    img = Image.open("./VAIPE_P_0_0.jpg")
    np_img = np.array(img)
    outputs = segmentation.segmented_pills(np_img)
    boxes = outputs["instances"].to("cpu").pred_boxes
    masks = outputs["instances"].to("cpu").pred_masks
    for i in range(len(boxes)):
        x1,y1,x2,y2 = boxes.tensor[i]
        x1,y1,x2,y2 = math.floor(x1),math.floor(y1),math.floor(x2),math.floor(y2)
        moi = masks[i][y1:y2,x1:x2].squeeze()
        seg_img = np_img[y1:y2,x1:x2]
        seg_img = seg_img * np.array(moi.unsqueeze(2))

        data = torch.tensor(np_img).unsqueeze(0).to(device)        
        with torch.no_grad():
        #   val_output = model(data).argmax(dim=1)
            val_output = model(data)
            print(val_output)
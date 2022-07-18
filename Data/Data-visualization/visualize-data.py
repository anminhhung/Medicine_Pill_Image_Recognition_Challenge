import cv2 
import json 
import os 
import numpy as np
from PIL import Image, ImageDraw, ImageFont
current_dir = os.path.abspath(os.getcwd())
print(os.path.abspath(os.getcwd()))
pill_pres_map_json_file = current_dir + "/public_train/pill_pres_map.json"
pres_dir = current_dir + '/public_train/prescription'
pill_dir = current_dir+ '/public_train/pill'

# read pill_pres_map.json
f = open(current_dir+'/public_train/pill_pres_map.json')
data = json.load(f)
print("data[1]:", data[1])

dict_label_pres = {}

def write_text(image, result_text, point):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # font_text = ImageFont.truetype("arial.ttf", 30)

    font_text = ImageFont.truetype(current_dir+"/font/SVN-Arial-Regular.ttf", 15)

    draw =  ImageDraw.Draw(pil_img)
    point = (point[0], int(point[1]-15))
    draw.text(point, result_text, (255, 0, 0), font=font_text)
    cv2_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return cv2_img

print("[INFO] Processing......!")
for obj in data:
    for key, value in obj.items():
        # print("key:", key)
        # print("value:", value)
        if key == 'pres':
            pres_json_file = value
            pres_img_file = value.split(".")[0] + ".png"

            pres_json_path = os.path.join(pres_dir, 'label', pres_json_file)
            pres_img_path = os.path.join(pres_dir, "image", pres_img_file)

            print("pres_json_path: ", pres_json_path)
            # json 
            json_f = open(pres_json_path)
            pres_data = json.load(json_f)
            # image
            pres_image = cv2.imread(pres_img_path)

            for obj_bbox in pres_data:
                id = obj_bbox["id"]
                text = obj_bbox["text"]
                label = obj_bbox["label"]
                bbox = obj_bbox["box"] 
                if label not in dict_label_pres:
                    dict_label_pres[label] = 0
                else:
                    dict_label_pres[label] = dict_label_pres[label] + 1

                # visualize
                pres_image = cv2.rectangle(pres_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 1)
                text_bbox = "id: {}, text: {}, label: {}".format(id, text, label)
                # pres_image = cv2.putText(pres_image, text_bbox, (bbox[0], bbox[1]), \
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # pres_image = Image.fromarray(cv2.cvtColor(pres_imgage, cv2.COLOR_BGR2RGB))
                pres_image = write_text(pres_image, text_bbox, (bbox[0], bbox[1]))

            pres_image = cv2.resize(pres_image, (1024, 1024))
            cv2.imshow("demo", pres_image)
            cv2.waitKey(0)
        else:
            for pill_json_file in value:
                pill_img_file = pill_json_file.split(".")[0] + ".jpg"

                pill_json_path = os.path.join(pill_dir, 'label', pill_json_file)
                pill_img_path = os.path.join(pill_dir, "image", pill_img_file)

                 # json 
                json_f = open(pill_json_path)
                pill_data = json.load(json_f)
                # image
                pill_image = cv2.imread(pill_img_path)

                for obj_bbox in pill_data:
                    x = obj_bbox["x"]
                    y = obj_bbox["y"]
                    h = obj_bbox["h"]
                    w = obj_bbox["w"]
                    label = obj_bbox["label"] 

                    bbox = [x, y, x+w, y+h]
                    # visualize
                    pill_image = cv2.rectangle(pill_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 5)
                    text_bbox = "label: {}".format(label)
                    
                    # pill_image = write_text(pill_image, text_bbox, (bbox[0], bbox[1]))
                    pill_image = cv2.putText(pill_image, text_bbox, (bbox[0], bbox[1]), 
                                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 10, cv2.LINE_AA)
                                        
                    

                pill_image = cv2.resize(pill_image, (1024, 1024))
                cv2.imshow('demo', pill_image)
                cv2.waitKey(0)

print("[INFO] Done!")
print(dict_label_pres)

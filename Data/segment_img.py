import json
import cv2
import numpy as np
import uuid

def get_segmented_imag(img_path,item_annotation,bbox):
  it = iter(item_annotation)
  arr = [*zip(it,it)]

  image = cv2.imread(img_path)

  mask = np.zeros(image.shape, dtype=np.uint8)
  roi_conners = np.array([arr], dtype=np.int32)
  channel_count = image.shape[2]
  ignore_mask_color = (255,) * channel_count
  cv2.fillPoly(mask,roi_conners,ignore_mask_color)
  masked_image = cv2.bitwise_and(image,mask)
  x,y, w, h = bbox
  x = int(x)
  y = int(y)
  w = int(w)
  h = int(h)
  cropped_masked_image = masked_image[y:y + h, x:x + w].copy()
  file_name = uuid.uuid4().hex + '.jpg'
  cv2.imwrite(f'./segmented_data/{file_name}',cropped_masked_image)


if __name__ == "__main__":

  f = open('./label.json')
  data = json.load(f)

  for obj in [item for item in data['annotations']]:
    item_annotation = obj['segmentation'][0]
    bbox = obj['bbox']
    img_id = obj['image_id']
    img = next(item2 for item2 in data['images'] if item2["id"] == img_id)
    img_path = './pill_images/' + img['file_name']
    print(img_path)
    get_segmented_imag(img_path,item_annotation,bbox)
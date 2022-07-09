###
### Convert AN image HBBs anno to OBBs anno using pretrained Mask RCNN segmentation map
###

import cv2
import numpy as np

def get_mask(roi):
    """
    Get mask of roi from Mask RCNN
    File txt of pixels of each instances
    """
    mask=[] # Get annotation file of mask of the roi (Can archive it to 1 folder)
    return mask

def img_convert(img,src_file_name,des_file_name):
    """
    img: an image
    src_file_name: annotation file contains many HBBs
    des_file_name: final DOTA style annotation file contains many OBBs
    """

    HBBs=[]
    """
    (wait for data)
    Extract HBBs information from src_file_name
    Append them into a list HBBs=[(x,y,w,h,category)]
    """

    for HBB in HBBs:
        x,y,w,h=HBB[0],HBB[1],HBB[2],HBB[3]
        temp=img[:].copy() # Extract roi HBB from img (need confirmation)
        label_name=get_mask(temp)# Retrieve mask of roi temp from Mask RCNN model
        result = help_convert(temp,label_name)
        # ............... more process
        # Projection the bbox according to HBB coordination !!!
        # Write result on des_file_name

def help_convert(img, label_name):
    # Process mask label file
    file = open(label_name, 'r')
    Lines = file.readlines()
    for i in range(len(Lines)):
        Lines[i]=Lines[i].strip()
    # Process mask of 1 roi
    mask=img.copy()
    index=0
    result=[]
    while(index<=len(Lines)-1):
        mask[:,:]=0
        if(Lines[index][0]=='I'):
            index+=1
            continue
        else:
            while(index<=len(Lines)-1 and Lines[index][0]!='I'):
                line=Lines[index].split()
                line[0]=int(line[0])
                line[1]=int(line[1])
                mask[line[0],line[1]]=255
                index+=1
            # Already have mask. Let's interpolate
            ret, thresh = cv2.threshold(mask, 127, 255, 0)
            contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            tmp=0
            for i in range(len(contours)):
                if(len(contours[i])>len(contours[tmp])):
                    tmp=i
            rect = cv2.minAreaRect(contours[tmp])
            (x,y),(w,h), a = rect
            box = cv2.boxPoints(rect)
            box = np.int0(box) #turn into ints
            result.append([len(contours[tmp]),box])
    maxi=0
    for i in range(len(result)):
        if result[i][0]>=result[maxi][0]:
            maxi=i
    return (result[maxi][1]) # tensor [[],[],[],[]] ordered as DOTA style
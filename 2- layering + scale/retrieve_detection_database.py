# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:15:39 2021

@author: elias
"""

import os
from openpyxl import load_workbook #to load existing workbooks
from openpyxl import Workbook #to make new workbooks


def load_results(result_path):
    detection_workbook = load_workbook(result_path) #calculation workbook
    detection_sheet = detection_workbook["Detection_Results"] #Parameters sheet
    
    result_dict = {"Result": ["Value", "Description"]}
    
    #detection_dict = {-1 : {"image_path": 0, "xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0, "label": 0, "confidence": 0, "x_size": 0, "y_size": 0}}
    detection_dict = {}
    
    i = 1
    #loop through sheet and fill dictionary
    for image, image_path, xmin, ymin, xmax, ymax, label, confidence, x_size, y_size in zip(detection_sheet["A"], 
                                                                                            detection_sheet["B"], 
                                                                                            detection_sheet["C"], 
                                                                                            detection_sheet["D"], 
                                                                                            detection_sheet["E"], 
                                                                                            detection_sheet["F"], 
                                                                                            detection_sheet["G"], 
                                                                                            detection_sheet["H"], 
                                                                                            detection_sheet["I"], 
                                                                                            detection_sheet["J"]
                                                                                            ):
        
        detection_dict[i] = {"image": image.value,
                             "image_path": image_path.value, 
                             "xmin": xmin.value, 
                             "ymin": ymin.value, 
                             "xmax": xmax.value, 
                             "ymax": ymax.value, 
                             "label": label.value, 
                             "confidence": confidence.value, 
                             "x_size": x_size.value, 
                             "y_size": y_size.value
                             }
        
        i += 1
    
    #print(detection_dict)
    return detection_dict
    

def retrieve_matching(detection_dict, label_index, raw_plan_path):
    #image
    image_name = os.path.split(raw_plan_path)[1]
    
    image_dict = {key:value for key,value in detection_dict.items() if value["image"] == image_name}
    #print(image_dict)
    
    #get max confidence value
    max_confidence = max(float(d['confidence']) for d in image_dict.values() if d["label"] == label_index)
    #print(max_confidence)
    
    max_confidence_bbox = []
    #loop through image dict and get values related to max confidence
    for d in image_dict.values():
        if d['confidence'] == max_confidence:
            max_confidence_bbox = [d['xmin'], d['ymin'], d['xmax'], d['ymax']]
            break
    
    #print(max_confidence_bbox)
    
    #get centroid of rectangle (y coordinate before x, because it is a matrix)
    pt_x = int( abs( max_confidence_bbox[2] - max_confidence_bbox[0] ) / 2 + max_confidence_bbox[0] ) #integer because matrix
    pt_y = int( abs( max_confidence_bbox[3] - max_confidence_bbox[1] ) / 2 + max_confidence_bbox[1] )
    max_confidence_point = [pt_y, pt_x]
    
    return max_confidence, max_confidence_bbox, max_confidence_point

"""
#load results
detection_dict = load_results("./plans - test05 (scale)/Detection_Results.xlsx")
#print(list(detection_dict.items())[0])

#retrieve matching rectangle/centroid with max confidence
max_confidence, max_confidence_bbox, max_confidence_point = retrieve_matching(detection_dict, 0, "./plans - test05 (scale)/31835886_z1_raw_plan.png")

print(max_confidence)
print(max_confidence_bbox)
print(max_confidence_point)
"""
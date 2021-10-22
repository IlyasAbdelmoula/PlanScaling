# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 14:45:44 2021

@author: admin
"""
import cv2 
import numpy as np
import copy

def check_background_condition(image_crop, active_coordinate_id, background_colour):
    if len(image_crop.shape) != 2: return True
    if image_crop.shape[0] < 1 or image_crop.shape[1] < 1: return True
    
    #condition
    #xmin
    print(image_crop.shape)
    if active_coordinate_id == 0:
        condition = np.all(image_crop[:, 0] == background_colour)
    #ymin
    if active_coordinate_id == 1:
        condition = np.all(image_crop[0,:] == background_colour)
    #xmax
    if active_coordinate_id == 2:
        condition = np.all(image_crop[:, -1] == background_colour)
    #ymax
    if active_coordinate_id == 3:
        condition = np.all(image_crop[-1,:] == background_colour)
    
    return condition

def image_crop_active_coordinate(image, single_prediction, active_coordinate_id, active_new_value):
    
    if active_coordinate_id == 0: #xmin
        image_crop = image[single_prediction[1]:single_prediction[3], active_new_value:single_prediction[2]]
    elif active_coordinate_id == 1: #xmin
        image_crop = image[active_new_value:single_prediction[3], single_prediction[0]:single_prediction[2]]  
    elif active_coordinate_id == 2: #xmin
        image_crop = image[single_prediction[1]:single_prediction[3], single_prediction[0]:active_new_value]
    elif active_coordinate_id == 3: #xmin
        image_crop = image[single_prediction[1]:active_new_value, single_prediction[0]:single_prediction[2]]   
    
    return image_crop

def image_crop_by_prediction(image, single_prediction):
    image_crop = image[single_prediction[1]:single_prediction[3], single_prediction[0]:single_prediction[2]] 
    return image_crop

def get_growth_limit(img_path, single_prediction, increment, background_colour): #255=white
    
    image_initial = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image_initial = cv2.cvtColor(image_initial, cv2.COLOR_RGB2GRAY)
    
    image = cv2.threshold(image_initial, 120, 255, cv2.THRESH_BINARY)[1]
    
    
    new_single_prediction = copy.deepcopy(single_prediction)

    for active_coordinate_id in range(4):
        print("current coordinate ID: {}".format(active_coordinate_id))
        active_coordinate = single_prediction[active_coordinate_id]
        new_active_coordinate = active_coordinate
        
        #yolo.detect_image output: out_prediction.append([left, top, right, bottom, c, score])
        
        if active_coordinate_id == 0 or active_coordinate_id == 1: 
            current_increment = - increment
        
        
        if active_coordinate_id == 2 or active_coordinate_id == 3: 
            current_increment = increment
        
        #else if active_coordinate_id == 2 or active_coordinate_id == 3: increment = increment
        
        #first loop
        #crop_img = img[y:y+h, x:x+w]
        image_crop = image[single_prediction[1]:single_prediction[3], single_prediction[0]:single_prediction[2]]
        print(image_crop.shape)
        #grow and check
        check = check_background_condition(image_crop, active_coordinate_id, background_colour)
        
        if check:
            continue
        
        while not check:
            new_active_coordinate = new_active_coordinate + current_increment
            #check shape to avoid loop
            previous_shape = image_crop.shape
            
            #crop a new image
            image_crop = image_crop_active_coordinate(image, single_prediction, active_coordinate_id, new_active_coordinate)
            
            #check the current shape if it is the same as previous, break
            if image_crop.shape == previous_shape:
                check = True
            else:
                check = check_background_condition(image_crop, active_coordinate_id, background_colour)
            
            if check: new_single_prediction[active_coordinate_id] = new_active_coordinate
    
    return new_single_prediction

"""
image_path = "Ic_RD0602_sommaire.png"
single_prediction = [515, 1093, 1806, 2209, 0, 0.89]

image_initial = im = cv2.imread(image_path, cv2.IMREAD_COLOR)
image_initial = cv2.cvtColor(image_initial, cv2.COLOR_RGB2GRAY)

image_crop = image_crop_by_prediction(image_initial, single_prediction)

print(image_crop.shape)


growth_limit_prediction = get_growth_limit(image_path, single_prediction, 20, 255)

image_crop_growth = image_crop_by_prediction(image_initial, growth_limit_prediction)


cv2.imwrite("image_crop_by_prediction.png", image_crop)
cv2.imwrite("image_crop_growth.png", image_crop_growth)

cv2.imshow("image_crop_by_prediction", image_crop)
cv2.imshow("image_crop_growth", image_crop_growth)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
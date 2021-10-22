# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 13:01:41 2021

@author: elias
"""

import cv2
import numpy as np
import blend_modes
import math

import glob #to retrieve filtered filename lists from directory

from matplotlib import pyplot as plt

import retrieve_detection_database


def detect_ORB(query_image_path, train_image_path):
    img1 = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE) # queryImage
    img2 = cv2.imread(train_image_path, cv2.IMREAD_GRAYSCALE) # trainImage
    # Initiate SIFT detector
    orb = cv2.ORB_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance) 
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
    
    plt.imshow(img3),plt.show()

def detect_ORB2(query_image_path, train_image_path):
    img1 = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE) # queryImage
    img2 = cv2.imread(train_image_path, cv2.IMREAD_GRAYSCALE) # trainImage
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance) 
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
    
    plt.imshow(img3),plt.show()

def detect_tut(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    cv2.imshow("image", img)

def extract_door_openings(wall_path, close_wall_path):
    #load images
    wall_img = cv2.imread(wall_path, cv2.IMREAD_GRAYSCALE)
    close_wall_img = cv2.imread(close_wall_path, cv2.IMREAD_GRAYSCALE)
    
    #get openings
    all_openings = close_wall_img - wall_img
    #all_openings = cv2.bitwise_not(all_openings) #invert
    
    contour_img = np.zeros(all_openings.shape)
    bbox_img = np.zeros(all_openings.shape)
    
    #find contours
    contours, hierarchy = cv2.findContours(all_openings, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for cnt in contours:
        #draw contour
        cv2.drawContours(contour_img, [cnt], -1, 0.4, thickness=1) #colour 0-1 because image subtracted
        
        #get rotated bounding box
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box=np.int0(box)
        #draw bounding box
        cv2.drawContours(bbox_img, [box], -1, 0.4, thickness=1) #colour 0-1 because image subtracted
        print(box)
        print(box[0][0])
        print(box[0][1])
        bounding_boxes.append(box)
    #bbox_img = cv2.bitwise_not(bbox_img) #invert
    #cv2.drawContours(bbox_img, [bounding_boxes[5]], -1, 100, thickness=2)
    
    #make a background
    white_background = np.zeros(all_openings.shape)
    white_background.fill(255)
    
    cv2.imshow("all_openings", all_openings)
    #cv2.imshow("contour_img", contour_img)
    #cv2.imshow("bbox_img", bbox_img)
    #cv2.imshow("white_background", white_background)
    
    return bounding_boxes, bbox_img, white_background
   
def get_pt_dist(pt1, pt2):
    delta_x = abs(pt1[1] - pt2[1])
    delta_y = abs(pt1[0] - pt2[0])
    dist = math.sqrt(delta_x**2 + delta_y**2)
    
    return dist

def get_confidence_pt_dist_squared(pt1, pt2):
    delta_x = abs(pt1[0] - pt2[1]) #bug #x and y of confidence poiint are inverted
    delta_y = abs(pt1[1] - pt2[0])
    dist_squared = delta_x**2 + delta_y**2
    
    return dist_squared   
    
def get_box_pixel_length(box):
    dim01 = get_pt_dist(box[0], box[1])
    #print(dim01)
    dim02 = get_pt_dist(box[1], box[2])
    #print(dim02)
    
    if dim01 > dim02: 
        return dim01
    else: 
        return dim02

def get_bbox_centroid(bbox):
    #get diagonal points
    pt_diag_1 = bbox[0]
    pt_diag_2 = bbox[2]
    
    centroid_y = int( abs(pt_diag_2[0] - pt_diag_1[0]) / 2 + min(pt_diag_1[0], pt_diag_2[0]) )
    centroid_x = int( abs(pt_diag_2[1] - pt_diag_1[1]) / 2 + min(pt_diag_1[1], pt_diag_2[1]) )
    
    centroid = (centroid_y, centroid_x)
    
    return centroid

def find_closest_bbox(bounding_boxes, point):
    #get bbox centroids
    bbox_centroids = []
    for bbox in bounding_boxes:
        centroid = get_bbox_centroid(bbox)
        bbox_centroids.append(centroid)
    
    #get squared distances to point
    dists_squared = []
    for centroid in bbox_centroids:
        dist = get_confidence_pt_dist_squared(point, centroid)
        #print(dist)
        dists_squared.append(dist)
    #print(dists_squared)
    #loop through distances, bboxes and retrieve the one with min distance
    min_dist = min(dists_squared)
    found = False
    for dist, bbox in zip(dists_squared, bounding_boxes):
        if dist == min_dist:
            print("closest distance = {}".format(dist))
            closest_bbox = bbox
            found = True
            break
        if found: break
    
    return closest_bbox
                
def highlight_bbox(bbox_img, bbox, confidence_point, label, thickness):
    
    cv2.drawContours(bbox_img, [bbox], -1, 1, thickness)
    cv2.circle(bbox_img, (confidence_point[1], confidence_point[0]), thickness, 0.6, thickness) #bug #x and y of confidence_point are inverted
    
    pt_label = (confidence_point[1] + thickness + 1, confidence_point[0] + thickness + 1)
    label = str(round(label, 2))
    
    cv2.putText(bbox_img,str(label),pt_label, cv2.FONT_HERSHEY_PLAIN, 0.8, 0.6, 1,cv2.LINE_AA)
    #bbox_img = ~bbox_img #invert
    cv2.imshow("opening bounding boxes", bbox_img)
    
    return bbox_img

def get_pixel_scale(length_pixel, length_meter, pixel_line_int):
    scale = length_pixel/length_meter
    scale_int = pixel_line_int / scale
    print("scale: 1 m = {} pixels".format(scale))
    print("scale integer: {} m = {} pixels".format(scale_int, pixel_line_int))
    return scale, scale_int, pixel_line_int

def get_int_scale(scale, pixel_range_start, pixel_range_end, tolerance):


    pixel_range = [i for i in range(pixel_range_end+1) if i >= pixel_range_start]
    #loop through range of pixel length
    for length_pixel in pixel_range:
        #test the meter length if integer
        length_meter = round(length_pixel / scale, tolerance)
        if length_meter.is_integer():
            #if so, return the values
            print("scale integer: {} m = {} pixels".format(length_meter, length_pixel))
            return length_meter, length_pixel
    
    #if nothing return scale as is
    return 1, scale

def draw_line(image, length_meter, length_pixel, margin, thickness, colour, font_scale, top=False):
    
    if top:
        pt0 = (margin, margin)
        pt1 = (pt0[0] + length_pixel, pt0[1] + thickness)
        pt_label = (pt1[0] + thickness, pt1[1] + thickness)
    else:
        pt0 = (margin, image.shape[0]-margin)
        pt1 = (pt0[0] + length_pixel, pt0[1] - thickness)
        pt_label = (pt1[0] + thickness, pt1[1] + thickness)
    
    
    
    #cv2.line(image, pt0, pt1, colour, thickness)
    cv2.rectangle(image, pt0, pt1, colour, thickness=-1)
    
    cv2.putText(image,str(length_meter)+"m",pt_label, cv2.FONT_HERSHEY_PLAIN, font_scale, colour,1,cv2.LINE_AA)
    
    #cv2.imshow("line", image)
    
    return image

def draw_scale_lines(image, scale, scale_int, pixel_line_int):
    #lower line (1m)
    draw_line(image,length_meter=1, length_pixel=int(scale), margin=5, thickness=3, colour=0, font_scale=0.8, top=False)
    
    #upper line (int)
    draw_line(image, length_meter=round(scale_int, 3), length_pixel=pixel_line_int, margin=5, thickness=3, colour=0, font_scale=0.8, top=True)
    
    cv2.imshow("scale_lines", image)
    
    return image
    
    
def draw_scale_lines_obsolete(image, scale, scale_int, pixel_line_int, tolerance, margin, thickness, colour, font_scale):
    #scale_label = str(round(scale, tolerance)) + " m"
    scale_int_round = round(scale_int, tolerance)
    
    #1m scale
    draw_line(image,
              colour,
              font_scale,
              length_meter=1, 
              length_pixel=int(scale), 
              margin=margin, 
              thickness=thickness, 
              top=False)
    
    #pixel_int scale
    draw_line(image,
              colour,
              font_scale,
              length_meter=scale_int_round, 
              length_pixel=pixel_line_int, 
              margin=margin, 
              thickness=thickness, 
              top=True)
    
    cv2.imshow("scale lines", image)

def save_image(image, base_path, output_suffix_ext):
    save_path = base_path + output_suffix_ext
    cv2.imwrite(save_path, image)

def get_base_path(source_path, suffix_to_remove):
    path = source_path.rstrip(suffix_to_remove)
    ##print(path)
    return path

#file lists
filelist_close_wall = glob.glob("./1- plans_test (output from floor plan detection)/*_z0_close_wall.png")
filelist_wall = glob.glob("./1- plans_test (output from floor plan detection)/*_z0_wall.png")
filelist_raw_plan = glob.glob("./1- plans_test (output from floor plan detection)/*_z1_raw_plan.png")

#filelist_raw_plans = glob.glob("./plans/*.jpg")

if len(filelist_close_wall) == len(filelist_wall) and len(filelist_close_wall) == len(filelist_raw_plan):
    
    #retrieve door database
    detection_dict = retrieve_detection_database.load_results("./1- plans_test (output from floor plan detection)/Detection_Results.xlsx")
    
    #loop through plans
    for close_wall_path, wall_path, raw_plan_path in zip(filelist_close_wall, 
                                                    filelist_wall, 
                                                    filelist_raw_plan):
        
        #--//OPENINGS AND THEIR BOUNDING BOXES//--
        
        #get base path
        base_path = get_base_path(close_wall_path, "_z0_close_wall.png")

        #get openings and their bounding boxes
        opening_bboxes, bbox_img, white_background = extract_door_openings(wall_path, close_wall_path)
        
        
        #--//SELECT BEST MATCHING BOX
        #select best door box
        max_confidence, max_confidence_bbox, max_confidence_point = retrieve_detection_database.retrieve_matching(detection_dict, 0, raw_plan_path)
        print(max_confidence, max_confidence_bbox, max_confidence_point)
        
        #select best door box (closest)
        matching_bbox = find_closest_bbox(opening_bboxes, max_confidence_point)
        print(matching_bbox)
        
        #update bbox_image
        bbox_img = highlight_bbox(bbox_img, matching_bbox, max_confidence_point, max_confidence, 2)
        
        #--//LENGTH AND SCALE CALCULATION//--
        door_dim = 0.9
        #get length
        door_pixel_length = get_box_pixel_length(matching_bbox)
        #print(door_pixel_length)
        
        #calculate scale
        scale, scale_int, pixel_line_int = get_pixel_scale(door_pixel_length, door_dim, pixel_line_int=50)
        
        #--//DRAW SCALE LINES//--
        """
        draw_scale_lines(white_background,
                         scale=scale, 
                         scale_int=scale_int,
                         pixel_line_int=pixel_line_int, 
                         tolerance=3, 
                         margin=5, 
                         thickness=3, 
                         colour=0, 
                         font_scale=0.6)
        """
        
        #scale lines on white
        scale_lines = draw_scale_lines(white_background, scale, scale_int, pixel_line_int)
        
        #scale lines on raw
        raw_plan = cv2.imread(raw_plan_path, cv2.IMREAD_GRAYSCALE)
        scale_lines_plan = draw_scale_lines(raw_plan, scale, scale_int, pixel_line_int)
        
        #--//SAVE IMAGES//--
        save_image(scale_lines, base_path, "_z0_scale.png")
        save_image(scale_lines_plan, base_path, "_z1_raw_plan_scale.png")
        save_image(bbox_img*255, base_path, "_z0_scale_openings.png") # *255 because all values are between 0-1
    
    #keybindings for preview
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
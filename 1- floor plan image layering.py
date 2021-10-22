# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 00:49:18 2021

@author: elias
"""

import cv2
import numpy as np
import blend_modes

import glob #to retrieve filtered filename lists from directory

import PIL
import tifffile

def get_walls_slab(closed_walls, open_walls):
    # READ IMAGES
    im_in = cv2.imread(closed_walls, cv2.IMREAD_GRAYSCALE)
    im_in2 = cv2.imread(open_walls, cv2.IMREAD_GRAYSCALE)
    
    #--//GET THE SLAB IMAGE (plan footprint)//--
    
    # INVERT WALLS IMAGES
    #METHOD01: Threshold
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.
    th, im_walls = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)
    ##print (im_walls)
    #METHOD02: Invert
    im_walls_open = cv2.bitwise_not(im_in2) #invert
    
    #FLOODFILLING (The background)
    # Copy the thresholded image.
    im_floodfill = im_walls.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_walls.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 220)
    
    # Combine the two images to get the floorpaln footprint
    im_subtracted = im_walls - im_floodfill
    
    #threshold then invert to get the footprint in black and the background in white
    th, im_subtracted_threshold = cv2.threshold(im_subtracted, 1, 250, cv2.THRESH_BINARY_INV)
    im_slab = cv2.bitwise_not(im_subtracted_threshold) #invert
    
    #--//OVERLAY walls on slab//--
    '''
    #test with a white image
    white = np.zeros(im_footprint.shape, dtype=np.uint8)
    white.fill(255) # or img[:] = 255
    
    white_rgba = cv2.cvtColor(white,cv2.COLOR_GRAY2RGBA)
    '''
    #cv2.floodFill(im_overlayed, mask, (0,0), 0)
    
    #for blendmodes to work, convert grayscale images to rgba
    im_slab_rgba = cv2.cvtColor(im_slab,cv2.COLOR_GRAY2RGBA)
    im_walls_open_rgba = cv2.cvtColor(im_walls_open,cv2.COLOR_GRAY2RGBA)
    
    im_overlayed = blend_modes.multiply(im_walls_open_rgba.astype(np.float32), im_slab_rgba.astype(np.float32), 0.1)
    
    im_overlayed_grayscale = cv2.cvtColor(im_overlayed.astype(np.uint8),cv2.COLOR_RGBA2GRAY)
    
    #--//OVERLAY walls on slabs for output (light walls colour)//--
    #white background
    white = np.zeros(im_slab_rgba.shape, dtype=np.uint8)
    white.fill(255) # or img[:] = 255
    #○im_walls_open_light = cv2.addWeighted(im_walls_open_rgba, 0.8, im_slab_rgba, 0.2, gamma=0)
    #light walls
    im_walls_open_light = cv2.addWeighted(white, 0.9, im_walls_open_rgba, 0.1, gamma=0)
    #walls + slab
    im_overlayed_light = cv2.addWeighted(im_walls_open_light, 0.9, im_slab_rgba, 0.1, gamma=0)
    im_overlayed_light = cv2.cvtColor(im_overlayed_light.astype(np.uint8),cv2.COLOR_RGBA2GRAY)
    
    #im_overlayed_light = blend_modes.multiply(im_walls_open_rgba.astype(np.float32), im_slab_rgba.astype(np.float32), 0.1)
    
    #--//OVERLAY walls on slabs for output (light slabs for post processing)
    im_slab_light = cv2.addWeighted(white, 0.9, im_slab_rgba, 0.1, gamma=0)
    im_slab_light = cv2.cvtColor(im_slab_light.astype(np.uint8),cv2.COLOR_RGBA2GRAY)
    
    # Display images.
    '''
    cv2.imshow("Thresholded walls Image", im_walls)
    cv2.imshow("Floodfilled walls Image", im_floodfill)
    #cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    cv2.imshow("walls - floodfilled", im_subtracted)
    #cv2.imshow("Foreground threshold", im_subtracted_threshold)
    cv2.imshow("Slab", im_slab)
    '''
    #cv2.imshow("Combined", im_overlayed.astype(np.uint8))
    cv2.imshow("walls_slab", im_overlayed_grayscale)
    cv2.imshow("walls_slab_light", im_overlayed_light)
    #cv2.waitKey(0)
    
    walls_slab = im_overlayed_grayscale

    slab_mask = im_subtracted_threshold
    walls_slab_light = im_overlayed_light
    
    slab_light = im_slab_light    
        
    return walls_slab, slab_mask, walls_slab_light, slab_light


def autocrop(image):
    bbox = cv2.boundingRect(image)

    x, y, w, h = bbox
    
    #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    
    #print(bbox)
    #print(x, y, w, h)
    foreground = image[y:y+h, x:x+w]
    
    cv2.imshow("Cropped", foreground)
    
    return bbox


def autocrop_threshold(image):
    th, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    autocrop(thresholded)


def autocrop_with_ref(image, reference_thresholded):
    bbox = autocrop(reference_thresholded)
    #print(bbox)
    x, y, w, h = bbox
    
    cropped = image[y:y+h, x:x+w]
    
    cv2.imshow("Cropped2", cropped)
    
    return cropped


def add_border(image_grayscale, border_width, border_colour=255):
    
    #cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    
    image_border = np.pad(image_grayscale, pad_width=border_width, mode='constant', constant_values=border_colour)
    
    #image_rgb = cv2.cvtColor(image_grayscale,cv2.COLOR_GRAY2RGB)
    #border_colour = (255, 255, 255)
    #image_border = cv2.copyMakeBorder(image_rgb,border,border,border,border,cv2.BORDER_CONSTANT, white) #not working in terms of colour choice, always gives black
    
    cv2.imshow("with borders", image_border)
    
    return image_border

def pad_to_square(image, colour_value):
    (a,b) = image.shape
    '''
    if a>b:
        padding=((0,0),(0,a-b))
    else:
        padding=((0,b-a),(0,0))
    
    image_square = np.pad(image, padding, mode='constant', constant_values=colour_value)
    '''
    colour = (colour_value, colour_value, colour_value)
    if a > b:
        delta = int((a-b)/2)
        #image_square = cv2.copyMakeBorder(image,delta,0,delta,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        image_square = cv2.copyMakeBorder(image,0,0,delta,delta,cv2.BORDER_CONSTANT,value=colour)
    else:
        delta = int((b-a)/2)
        image_square = cv2.copyMakeBorder(image,delta,delta,0,0,cv2.BORDER_CONSTANT,value=colour)
    #border_colour = (255, 255, 255)
    #image_border = cv2.copyMakeBorder(image_rgb,border,border,border,border,cv2.BORDER_CONSTANT, white) #not working in terms of colour choice, always gives black
    
    cv2.imshow("square", image_square)
        
    return image_square

def crop_and_square(image, crop_mask, base_path, output_suffix, border_width, save=False, border_colour=255):
    #crop plan to slab boundaries
    cropped = autocrop_with_ref(image, crop_mask)
    #add small margin
    border = add_border(cropped, border_width, border_colour)
    #pad to create a sq
    square = pad_to_square(border, border_colour)
    
    if save:
        #new path
        save_path = base_path + output_suffix + ".png"
        
        cv2.imwrite(save_path, square)
    
    return square
 

def get_base_path(source_path, suffix_to_remove):
    path = source_path.rstrip(suffix_to_remove)
    ##print(path)
    return path

def save_multipage_tiff(raw_plan, slab_light, walls_slab_light, base_path, suffix):
    save_path = base_path + suffix + ".tif"
    
    images = [raw_plan, slab_light, walls_slab_light]
    pages = []
    #convert images to PIL images
    for i in images:
        img = PIL.Image.fromarray(i)
        pages.append(img)
    
    #save the first page whill emùbedding the others
    pages[0].save(save_path, compression="tiff_deflate", save_all=True,
               append_images=pages[1:])
    #tifffile.imsave(save_path,pages)

#file lists
filelist_close_wall = glob.glob("./plans - test05 (crop raw walls as well)/*_close_wall.png")
filelist_wall = glob.glob("./plans - test05 (crop raw walls as well)/*_wall.png")
filelist_wall = [i for i in filelist_wall if i not in filelist_close_wall]
#filelist_raw_plans = glob.glob("./plans/*.jpg")

if len(filelist_close_wall) == len(filelist_wall): # and len(filelist_close_wall) == len(filelist_raw_plans):
    for close_wall_path, wall_path in zip(filelist_close_wall, filelist_wall):
        #get wall+slab representaitions
        walls_slab, slab_mask, walls_slab_light, slab_light = get_walls_slab(close_wall_path, wall_path)
        
        #get base path
        base_path = get_base_path(close_wall_path, "_close_wall.png")
        
        #crop, pad to square, save
        walls_slab = crop_and_square(walls_slab, slab_mask, base_path, "_z2_walls_slab", border_width=20, save=True)
        walls_slab_light = crop_and_square(walls_slab_light, slab_mask, base_path, "_z3_walls_slab_light", border_width=20, save=True)
        slab_light = crop_and_square(slab_light, slab_mask, base_path, "_z4_slab_light", border_width=20, save=True)
        
        #raw plan square + save
        raw_plan_path = base_path + ".jpg"
        raw_plan = cv2.imread(raw_plan_path, cv2.IMREAD_GRAYSCALE)
        raw_plan = crop_and_square(raw_plan, slab_mask, base_path, "_z1_raw_plan", border_width=20, save=True)
        
        #save multipage tiff for further process
        save_multipage_tiff(raw_plan, slab_light, walls_slab_light, base_path, "_z0_walls_merged")
        
        #original open/closed wall plans crop
        raw_close_wall = cv2.imread(close_wall_path, cv2.IMREAD_GRAYSCALE)
        raw_close_wall = crop_and_square(raw_close_wall, slab_mask, base_path, "_z0_close_wall", border_width=20, save=True, border_colour=0)
        raw_wall = cv2.imread(wall_path, cv2.IMREAD_GRAYSCALE)
        raw_wall = crop_and_square(raw_wall, slab_mask, base_path, "_z0_wall", border_width=20, save=True, border_colour=0)
        
        #save slab mask
        slab_mask_square = crop_and_square(slab_mask, slab_mask, base_path, "_z0_slab_mask", border_width=20, save=True, border_colour=0)
    
    print ("{} plans successfully processed".format(len(filelist_close_wall)))

else:
    print("nothing happens, dataset files are not matching!")
        
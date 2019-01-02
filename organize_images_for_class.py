
from PIL import Image
import os
import math
import numpy as np
import pandas as pd



def organize_images_for_class(class_name,total_use,standardized_image_path,training_set_size,download_path,basewidth):

    #create the dir for these
    standardized_image_class_train_path=standardized_image_path + "train/" + class_name + "/"
    standardized_image_class_validate_path = standardized_image_path + "validate/" + class_name + "/"
    if not os.path.exists(standardized_image_class_train_path):
        os.mkdir(standardized_image_class_train_path)
    if not os.path.exists(standardized_image_class_validate_path):
        os.mkdir(standardized_image_class_validate_path)

    #decide which images will be in training and which will be validation
    training_set_indices = np.random.choice(range(0,total_use),size=training_set_size,replace=False)
    is_training_set =  [x in training_set_indices for x in range(0,total_use)]

    #I don't think so. we'll probably have to just resize and crop
    i=-1
    for f in os.listdir(download_path + class_name):
        if (f.lower().endswith(".png") or f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")):
            if i+1>=total_use:
                break #we got enough, break.


            # try to open image; if we can't, skip it.
            try:
                img = Image.open(download_path + class_name + "/" + f)
            except IOError:
                print("error trying to access image " + class_name + "/" + f + ". Skipping...")
                continue

            i = i + 1  # count images we're actually using.

            print ((i, f))
            #crop first to get a constant aspect ratio.
            # now having set width to what we want, we need to crop the height if it is too large.
            # we are doing this based on a fixed width, aiming to get a
            new_diameter = min(img.size)
            crop_width = img.size[0]-min(img.size)
            crop_height = img.size[1] - min(img.size)
            crop_left = int(math.floor(crop_width/2))
            crop_right = int(math.floor(crop_width/2 + new_diameter))
            crop_top = int(math.floor(crop_height/2))
            crop_bottom = int(math.floor(crop_height/2 + new_diameter))
            img = img.crop((crop_left,crop_top,crop_right,crop_bottom))

            #resize
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize), Image.ANTIALIAS)


            #save image in either training or validation
            if is_training_set[i]:
                img.save(standardized_image_class_train_path + f)
            else:
                img.save(standardized_image_class_validate_path + f)

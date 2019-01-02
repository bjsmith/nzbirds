from google_images_download import google_images_download
from PIL import Image
import os
import math
import numpy as np
import pandas as pd
response = google_images_download.googleimagesdownload()
#set up the image downloader

#define the classes of objects we want to search for
bird_list=pd.read_csv("/Users/benjaminsmith/Documents/data/nzbirds/birdlist.csv")


#set the path to download to and make sure the dirs exist.
download_path = '/Users/benjaminsmith/Documents/data/nzbirds/test_exemplars70/'
standardized_image_path = '/Users/benjaminsmith/Documents/data/nzbirds/test_exemplars70_standardized/'
for dir_to_create in [standardized_image_path,standardized_image_path + "train/",standardized_image_path + "validate/"]:
    if not os.path.exists(dir_to_create):
        os.mkdir(dir_to_create)

basewidth=300

training_set_size=50
validation_set_size=20
total_use =training_set_size+validation_set_size
total_download=100#int(math.ceil(1.35*(total_use))) #get a few more than we need because some images are invalid.
np.random.seed(81231113)

def download_learning_class(class_keyterm, total_download, download_path):
    response.download({
        "keywords": class_keyterm + " bird",  # add "bird" to this to narrow search, can be helpful.
        "limit": total_download,
    # we don't want large database because perhaps we don't need it because we have a pre-trained classifier.
        "output_directory": download_path,
        "image_directory": class_keyterm,  # set where we're storing these.
        "type": "photo"  # we also want only photos
    })

def organize_images(class_keyterm,total_use):

    #create the dir for these
    standardized_image_class_train_path=standardized_image_path + "train/" + class_keyterm + "/"
    standardized_image_class_validate_path = standardized_image_path + "validate/" + class_keyterm + "/"
    if not os.path.exists(standardized_image_class_train_path):
        os.mkdir(standardized_image_class_train_path)
    if not os.path.exists(standardized_image_class_validate_path):
        os.mkdir(standardized_image_class_validate_path)

    #decide which images will be in training and which will be validation
    training_set_indices = np.random.choice(range(0,total_use),size=training_set_size,replace=False)
    is_training_set =  [x in training_set_indices for x in range(0,total_use)]

    #I don't think so. we'll probably have to just resize and crop
    i=-1
    for f in os.listdir(download_path + class_keyterm):
        if (f.lower().endswith(".png") or f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")):
            if i+1>=total_use:
                break #we got enough, break.


            # try to open image; if we can't, skip it.
            try:
                img = Image.open(download_path + class_keyterm + "/" + f)
            except IOError:
                print("error trying to access image " + class_keyterm + "/" + f + ". Skipping...")
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


for index, bird_row in bird_list.iterrows():
    print((bird_row["Name"], bird_row["Machine name"]))

    #now we loop through the classes and download
    download_learning_class(class_keyterm, total_download, download_path)

    organize_images(class_keyterm,total_use)

#resize


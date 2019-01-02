from google_images_download import google_images_download
import numpy as np
import pandas as pd
#from organize_images import *
response = google_images_download.googleimagesdownload()
#set up the image downloader


#set the path to download to and make sure the dirs exist.
download_path = '/Users/benjaminsmith/Documents/data/nzbirds/exemplars70/'

basewidth=200

training_set_size=50
validation_set_size=20
total_use =training_set_size+validation_set_size
total_download=100#int(math.ceil(1.35*(total_use))) #get a few more than we need because some images are invalid.
np.random.seed(81231113)

def download_learning_class(class_keyterm, total_download, download_path,image_dir=None):
    if(image_dir is None):
        image_dir=class_keyterm

    response.download({
        "keywords": class_keyterm + " bird",  # add "bird" to this to narrow search, can be helpful.
        "limit": total_download,
    # we don't want large database because perhaps we don't need it because we have a pre-trained classifier.
        "output_directory": download_path,
        "image_directory": image_dir,  # set where we're storing these.
        "type": "photo"  # we also want only photos
    })
#
# for index, bird_row in bird_list.iterrows():
#     class_keyterm = bird_row["Search term"]
#     #now we loop through the classes and download
#     download_learning_class(class_keyterm, total_download, download_path,image_dir=bird_row["Machine name"])
#
#     #organize_images(bird_row["Machine name"],total_use)
#
# #resize
#

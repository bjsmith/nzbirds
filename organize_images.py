
from PIL import Image
import os
import math
import numpy as np
import pandas as pd
from organize_images_for_class import *

bird_list=pd.read_csv("/Users/benjaminsmith/Documents/data/nzbirds/birdlist.csv")

download_path = '/Users/benjaminsmith/Documents/data/nzbirds/exemplars70/'


standardized_image_path = '/Users/benjaminsmith/Documents/data/nzbirds/exemplars70_standardized/'
for dir_to_create in [standardized_image_path,standardized_image_path + "train/",standardized_image_path + "validate/"]:
    if not os.path.exists(dir_to_create):
        os.mkdir(dir_to_create)

basewidth=200

training_set_size=50
validation_set_size=20
total_use =training_set_size+validation_set_size
total_download=100#int(math.ceil(1.35*(total_use))) #get a few more than we need because some images are invalid.
np.random.seed(8123118)



for index, bird_row in bird_list.iterrows():
    class_keyterm = bird_row["Search term"]
    #now we loop through the classes and download
    #download_learning_class(class_keyterm, total_download, download_path,image_dir=bird_row["Machine name"])

    if(os.path.exists(download_path + bird_row["Machine name"])):
        organize_images(bird_row["Machine name"],total_use,standardized_image_path,training_set_size,download_path,basewidth)

from google_images_download import google_images_download

from download_learning_class import *
#set up the image downloader

#define the classes of objects we want to search for
bird_list_all=pd.read_csv("/Users/benjaminsmith/Google Drive/ml-projects/data/nzbirds/birdlist.txt", encoding='utf_16',sep='\t')

bird_list = bird_list_all.loc[lambda df: df['Notes']=='Repeat again']
#set the path to download to and make sure the dirs exist.
download_path = '/Users/benjaminsmith/Google Drive/ml-projects/data/nzbirds/exemplars70/'

total_download=100#int(math.ceil(1.35*(total_use))) #get a few more than we need because some images are invalid.
np.random.seed(81231113)

for index, bird_row in bird_list.iterrows():
    class_keyterm = bird_row["Search term"].encode('utf-8')
    #now we loop through the classes and download
    download_learning_class(class_keyterm, total_download, download_path,image_dir=bird_row["Machine name"])

    #organize_images(bird_row["Machine name"],total_use)

#resize


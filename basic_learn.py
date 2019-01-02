from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys

num_classes=4

resnet_weights_path = '/Users/benjaminsmith/Documents/data/nzbirds/ResNet-50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
train_dir ='/Users/benjaminsmith/Documents/data/nzbirds/test_exemplars70_standardized/train/'
validation_dir ='/Users/benjaminsmith/Documents/data/nzbirds/test_exemplars70_standardized/validate/'

print ("creating net...")
bird_model = Sequential()
bird_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
bird_model.add(Dense(num_classes, activation='softmax'))
bird_model.layers[0].trainable = False
sys.stdout.write("...compiling net...")
bird_model.compile(optimizer="sgd",loss='categorical_crossentropy',metrics=['accuracy'])
sys.stdout.write("...Compiled. Getting images...")
#size of the images
image_dim = (200,200)

#data generator for training
train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                          horizontal_flip=True,
                                                     rotation_range=45,
                                                    width_shift_range=0.2,
                                                    height_shift_range=0.2)
#data generator for validation. just does pre-processing
validation_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)



train_generator = train_data_generator.flow_from_directory(directory = train_dir,
                                                     target_size=image_dim,
                                                     batch_size=10,
                                                     class_mode='categorical')

#should look at improving on the augmentation!

validation_generator = validation_data_generator.flow_from_directory(directory = validation_dir,
                                                     target_size=image_dim,
                                                     batch_size=10,
                                                     class_mode='categorical')

sys.stdout.write("...Fitting...")
fit_stats = bird_model.fit_generator(train_generator,
                                     epochs=2,
                                     steps_per_epoch=15,#if we weren't doing augmentation this would repeat each image 3 times.
                                     #because we're using augmentation, that is not the case.
                                     validation_data=validation_generator,
                                     validation_steps=1)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys

num_classes=54
#learns 54 different native and exotic birds in new zealand, using up to 70 exemplars for each. In reality we weren't able to get enough to get 70 in all classes.

resnet_weights_path = '/Users/benjaminsmith/Documents/data/nzbirds/ResNet-50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
train_dir ='/Users/benjaminsmith/Documents/data/nzbirds/exemplars70_standardized/train/'
validation_dir ='/Users/benjaminsmith/Documents/data/nzbirds/exemplars70_standardized/validate/'

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

#we have 55 classes with almost 50 images each in the training data
#that's 55*50=2750 images in the training set, although it's a few less than that because we don't have enough images in each set
#update: there are 2638 [training?] images in total.
#HOWEVER the horizontal flip doubles our effective images, and the rotation range and width and height shifts give us
#much more data, so theoretically we might have a much larger number than that.
#So with a batch size of 10 then we would theoretically ahve 275 steps per epoch. But we could have more (because we've augmented the data)
#and could also have less (because we might not have enough memory for that number.
#let's go up to a number equal to half of the dataset each time which is 138 images.

#epochs - first epoch achieves 0.2 validation accuracy, so probably we should train more than 1 epoch.
#Epoch 2: 0.5 validation accuracy.
#Epoch 3: 0.5
#Epoch 4: 0.6
#Epoch 5: 0.7

sys.stdout.write("...Fitting...")
fit_stats = bird_model.fit_generator(train_generator,
                                     epochs=6,
                                     steps_per_epoch=138,
                                     #maybe I should increase this to 275 images?
                                     validation_data=validation_generator,
                                     validation_steps=1)

#OK the validation steps is way off! It's only doing 10 each time!
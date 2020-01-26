"""
Run this program second
This will generate Xtest.npy, Xtrain.npy, Ytest.npy and Ytrain.npy
"""

import os
import numpy as np
import re
from PIL import Image as PImage

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def loadImages(path):
    # return array of images
    imagesList = os.listdir(path)
    loadedImages = []
    for image in imagesList:

        with PImage.open(path + image) as img:
            img = np.array(img)
            img = img.reshape((img.shape[0],img.shape[1],1))
            loadedImages.append(img)

    return loadedImages


traindir = "Train/"
testdir = "Test/"

# Get all subdirs of train dir
train_subdirs = [traindir + i + '/' for i in os.listdir(traindir)]
train_subdirs.sort(key=natural_keys) # Sorting character folder names by their number via Human/Natural Sorting

# Get all subdirs of test dir
test_subdirs = [testdir + i + '/' for i in os.listdir(testdir)]
test_subdirs.sort(key=natural_keys) # Sorting character folder names by their number via Human/Natural Sorting

train_images = []
test_images = []

for path in train_subdirs:
    print("Extracting characters from ",path)
    tr_imgs = loadImages(path)
    train_images.append(tr_imgs)

print("\n")

for path in test_subdirs:
    print("Extracting characters from ",path)
    te_imgs = loadImages(path)
    test_images.append(te_imgs)


print(len(train_images))
print(len(test_images))
print(len(train_images[0]))
print(len(test_images[0]))
print(type(train_images[0][0]))
print(type(test_images[0][0]))
print(train_images[0][0].shape)
print(test_images[0][0].shape)

# class labels are appended
# ytrain length is the total # of traing images
# i is the total number of characters
# j is the number of images for each type of character
Ytrain = []
for i in range(len(train_images)):
    for j in range(len(train_images[i])):
        Ytrain.append(i)
Ytrain = np.array(Ytrain)

#same as above
Ytest = []
for i in range(len(test_images)):
    for j in range(len(test_images[i])):
        Ytest.append(i)
Ytest = np.array(Ytest)


#here the input images are appended
Xtrain = []
for i in range(len(train_images)):
    for j in range(len(train_images[i])):
        Xtrain.append(train_images[i][j])
Xtrain = np.array(Xtrain)

Xtest = []
for i in range(len(train_images)):
    for j in range(len(test_images[i])):
        Xtest.append(test_images[i][j])
Xtest = np.array(Xtest)


print("Shape of Xtrain = ",Xtrain.shape)
print("Shape of Xtest = ",Xtest.shape)
print("Shape of Ytrain = ",Ytrain.shape)
print("Shape of Ytest = ",Ytest.shape)


Xtrain = Xtrain / 255
Xtest = Xtest / 255

np.save('Xtrain.npy', Xtrain)
np.save('Xtest.npy', Xtest)
np.save('Ytrain.npy', Ytrain)
np.save('Ytest.npy', Ytest)

"""
Run this program first. Extract input.zip before running.
Creates the Train and Test folders.
It is a linear split. Each character has 2000 images.
Train:Test is 1700:300 for each character.
"""
import os
from shutil import copyfile

source="input/Images/Images"
train="Train"
test="Test"

os.mkdir(train)
os.mkdir(test)

for i in os.listdir(source):
    os.mkdir(train+"/"+i)
    os.mkdir(test+"/"+i)
    count=0
    num=len(os.listdir(source+"/"+i))
    for j in os.listdir(source+"/"+i):
        if count<1700:
            copyfile(source+"/"+i+"/"+j, train+"/"+i+"/"+j)
        else:
            copyfile(source+"/"+i+"/"+j, test+"/"+i+"/"+j)
        count+=1

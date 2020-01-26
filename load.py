"""
Run this program fourth.
Make sure to change line 84 by changing the image location to the location of the image you want to test.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pydub import AudioSegment
from pydub.playback import play


def show_image(image):
    # Plots image
    #assert len(image.shape) == 3, "Image passed in is of incorrect shape"
    plt.imshow(image.squeeze())
    plt.show()


devchar = {0: "क",
           1: "ख",
           2:"ग",
           3:"घ",
           4:"ङ",
           5:"च",
           6:"छ",
           7:"ज",
           8:"झ",
           9:"ञ",
           10:"ट",
           11:"ठ",
           12:"ड",
           13:"ढ",
           14:"ण",
           15:"त",
           16:"थ",
           17:"द",
           18:"ध",
           19:"न",
           20:"प",
           21:"फ",
           22:"ब",
           23:"भ",
           24:"म",
           25:"य",
           26:"र",
           27:"ल",
           28:"व",
           29:"श",
           30:"ष",
           31:"स",
           32:"ह",
           33:"क्ष",
           34:"त्र",
           35:"ज्ञ",
           36:"०",
           37:"१",
           38:"२",
           39:"३",
           40:"४",
           41:"५",
           42:"६",
           43:"७",
           44:"८",
           45:"९",}


#loading the model
sess=tf.Session()
saver = tf.train.import_meta_graph('Models/my_model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('Models/./'))

graph = tf.get_default_graph()
tfX = graph.get_tensor_by_name("X:0")
tfT = graph.get_tensor_by_name("T:0")

Yish = graph.get_tensor_by_name("op_to_restore:0")

X_test = np.load('Xtest.npy')
Y_test = np.load('Ytest.npy')

img = cv2.imread("test1.jpg",0)

#Getting the bigger side of the imag
s = max(img.shape[0:2])

#Creating a dark square with NUMPY
f = np.full((s,s),255,np.uint8)

#Getting the centering position
ax,ay = (s - img.shape[1])//2,(s - img.shape[0])//2

#Pasting the 'image' in a centering position
f[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img

ret, bw_img = cv2.threshold(f,127,255,cv2.THRESH_BINARY)

bw_img = cv2.bitwise_not(bw_img)
print(bw_img.shape)

bw_img = cv2.resize(bw_img,(32,32))
print(len(bw_img.shape))
bw_img = bw_img.reshape(1,bw_img.shape[0],bw_img.shape[1],1)
show_image(bw_img)


logits = sess.run(Yish,feed_dict={tfX:bw_img})
preds = tf.argmax(logits,axis=1)
result=preds.eval(session=sess)
print("Predicted Label Class = ",result)
print("Predicted Character = ",devchar[result[0]])

sound = AudioSegment.from_mp3("audio/"+str(result[0])+".mp3")
play(sound)

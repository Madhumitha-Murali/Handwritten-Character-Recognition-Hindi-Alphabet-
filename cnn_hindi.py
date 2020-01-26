"""
Run this program third
Creates the model and runs the model on the test images.
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def y2indicator(Y):
    N = len(Y)
    K = len(set(Y))
    Y_ind = np.zeros((N,K))
    for i in range(N):
        Y_ind[i,Y[i]] = 1
    return Y_ind

def error_rate(targets,predictions):
    return np.mean(targets != predictions)

def convpool(X,W,b):
    conv_out = tf.nn.conv2d(X, W, strides = [1,1,1,1], padding = 'VALID')
    conv_out = tf.nn.bias_add(conv_out, bias = b)
    pool_out = tf.nn.max_pool(conv_out, ksize = [1,2,2,1], strides= [1,2,2,1], padding='VALID')
    return tf.nn.relu(pool_out)

def init_filter(shape):
    W = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[:-1]))
    return W.astype(np.float32)

def init_weight_and_bias(M1,M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)



def show_image(image):
    # Plots image
    assert len(image.shape) == 3, "Image passed in is of incorrect shape"
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


#training the model

X = np.load('Xtrain.npy')
Y = np.load('Ytrain.npy')

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, random_state=42)

Y_train_ind = y2indicator(Y_train)
Y_valid_ind = y2indicator(Y_valid)

# Gradient descent params

N = X_train.shape[0]
M = 500
K = len(set(Y_train))

epochs = 20
print_period = 10
batch_sz = 595
n_batches = N // batch_sz
reg = 0.001

# Initialise Variables
W1_shape = (5,5,1,10) # (HWFinFout)
W1_init = init_filter(W1_shape)
b1_init = np.zeros(W1_shape[-1],dtype=np.float32)

W2_shape = (5,5,10,30) # (HWFinFout)
W2_init = init_filter(W2_shape)
b2_init = np.zeros(W2_shape[-1],dtype=np.float32)

W3_init, b3_init = init_weight_and_bias(W2_shape[-1]*5*5,M)

W4_init, b4_init = init_weight_and_bias(M,K) # K 47 class labels

# Set Placeholders & Variables

tfX = tf.placeholder(dtype=tf.float32,shape=(None,32,32,1),name="X")
tfT = tf.placeholder(dtype=tf.float32,shape=(None,K),name="T")

W1 = tf.Variable(initial_value=W1_init,dtype=tf.float32)
b1 = tf.Variable(initial_value=b1_init,dtype=tf.float32)
W2 = tf.Variable(initial_value=W2_init,dtype=tf.float32)
b2 = tf.Variable(initial_value=b2_init,dtype=tf.float32)
W3 = tf.Variable(initial_value=W3_init,dtype=tf.float32)
b3 = tf.Variable(initial_value=b3_init,dtype=tf.float32)
W4 = tf.Variable(initial_value=W4_init,dtype=tf.float32)
b4 = tf.Variable(initial_value=b4_init,dtype=tf.float32)

#5x5x10, window of size 5x5, 10 feature maps, output will be 28x28x10 after convolution, 14x14x10 after max pooling
#trainable parameters = weights+bias=5*5*1*10+10=260
Z1 = convpool(tfX,W1,b1)

#5x5x30, window of size 5x5, 30 feature maps, output will be 10x10x30 after convolution, 5x5x30 after max pooling
#trainable parameters = weights+bias=5*5*10*30+30=7,530
Z2 = convpool(Z1,W2,b2)

Z2_shape = Z2.get_shape()
num_features=Z2_shape[1:].num_elements()
print(Z2_shape)
Z2f = tf.reshape(Z2,[-1,num_features]) # NHWC => N x (HWC)


#fully connected layer of size M=500
#trainable parameters = weights+bias=5*5*30*500+500=3,75,500
Z3 = tf.nn.relu(tf.add(tf.matmul(Z2f,W3),b3))

#fully connected layer of size K=47
#trainable parameters = weights+bias=500*47+47=23,547
Yish = tf.add(tf.matmul(Z3,W4),b4,name="op_to_restore")


cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tfT,logits=Yish))

regularizer = (tf.nn.l2_loss(W1) +
               tf.nn.l2_loss(W2) +
               tf.nn.l2_loss(W3) +
               tf.nn.l2_loss(W4))

# Add L2 regularisation
reg_cost = (cost + reg * regularizer)


train_op = tf.train.RMSPropOptimizer(learning_rate=1e-3,decay=0.99,momentum=0.9).minimize(reg_cost)
predict_op = tf.argmax(Yish,axis=1)

t0 = datetime.now()
costs = []

init = tf.global_variables_initializer()  #variables hold the values you told them to hold when you declare them

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        for j in range(n_batches):
            Xbatch = X_train[j*batch_sz:j*batch_sz+batch_sz]
            Ybatch = Y_train_ind[j*batch_sz:j*batch_sz+batch_sz]
            sess.run(train_op,feed_dict={tfX:Xbatch,tfT:Ybatch})

            if j % print_period == 0:
                prediction = np.zeros(len(X_valid))
                valid_cost = 0
                for k in range(len(X_valid)//batch_sz):
                    Xvbatch = X_valid[k*batch_sz:k*batch_sz+batch_sz]
                    Yvbatch = Y_valid_ind[k*batch_sz:k*batch_sz+batch_sz]
                    prediction[k*batch_sz:k*batch_sz+batch_sz] = sess.run(predict_op,feed_dict={tfX:Xvbatch})
                    batch_cost = sess.run(cost,feed_dict={tfX:Xvbatch,tfT:Yvbatch})
                    valid_cost += batch_cost
                error = error_rate(targets=Y_valid,predictions=prediction)
                print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, valid_cost, error))
                costs.append(valid_cost)

    print("Final Validation Accuracy = ", str(round((1-error)*100,3)) + " %")

    # Create directory if necessry to save model
    if os.path.isdir('Models') is False:
        os.mkdir('Models')

    saver.save(sess,'Models/my_model.ckpt')

    print("Elapsed time:", (datetime.now() - t0))
    plt.plot(costs)
    plt.show()


    # Load test data
    X_test = np.load('Xtest.npy')
    Y_test = np.load('Ytest.npy')

    n_batches_test = X_test.shape[0] // batch_sz # 58 batches

    # Get predictions for test set
    results = np.array([0]*X_test.shape[0])
    for b in range(n_batches_test):
        Xtbatch = X_test[b*batch_sz:b*batch_sz+batch_sz]
        logits = Yish.eval(feed_dict={tfX:Xtbatch})
        preds = tf.argmax(logits,axis=1)
        results[b*batch_sz:b*batch_sz+batch_sz] = preds.eval()

    print("Final Test Accuracy = ", str(round((1-error_rate(Y_test,results))*100,3)) + " %")

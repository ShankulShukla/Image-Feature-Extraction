
# Working improved dataset with images of pixel 480*640

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

# Open cv library
import cv2


# Changing to the directory where the images are located
import os
os.chdir(r'C:\Users\shankul\Downloads\sign data')


# Loading the images, resizing it to 128*128 pixels

df = pd.DataFrame()
for folder in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']:
    img_list = os.listdir(folder)
    print('Loading Folder - {} ...'.format(folder))
    for i,img in enumerate(img_list):
        inp = cv2.imread(os.path.join(folder,img))
        input_img = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
        inp_resize = cv2.resize(input_img,(128,128))
        inp_resize.resize(1,128*128)
        df = df.append(pd.DataFrame(inp_resize),ignore_index=True)


# Storing the dataframe into a csv file 
df.to_csv('data.csv',index=False)

# You can now load the csv file
df = pd.read_csv("data.csv")


# Making labels for the dataset

l = []
label = []
for folder in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']:
    img_list = os.listdir(folder)
    i = ord(folder)
    temp = [i-65]*len(img_list)
    l.append(len(img_list))
    label.append(temp)

# transforming them into a numpy array
label = np.concatenate(label) 

# checking for the dataframe shape 
print(df.shape)

# Creating classes to label dictionary 

ans = {}
for i in range(65,90):
    ans[i-65] = chr(i)
del ans[9]


# practically seeing the images with some index
index = 3509
plt.imshow(df.loc[index].values.reshape(128,128))
print(ans[label[index]])


# Ramdomly suffling the training samples

from sklearn.utils import shuffle
a, b = shuffle(df.values, label, random_state=0)
df_ = pd.DataFrame(a)


# normalizing the data in range(0 to 1)
df_ = df_ / 255.0


# spliting the dataset into training validation dataset

x_train, y_train, x_val, y_val = df_[:4500].values, b[:4500], df_[4500:].values, b[4500:]

# Hyperparameters
classes = 24
epochs = 100
bs = 12



def batch_generator(bs,x,y=None,test=False):
    for i in range(0,len(x),bs):
        if test:
            yield x[i:i+bs]
        else:
            yield x[i:i+bs], y[i:i+bs]

# Input placeholders 

X = tf.placeholder(dtype=tf.float32,name='X',shape=(None,128,128,1))
Y = tf.placeholder(dtype=tf.int32,name='Y',shape=(None))
keep_prob = tf.placeholder(tf.float32,name='keep_prob')


# Network Architecture and design
conv1 = tf.layers.conv2d(inputs=X,kernel_size=(5,5),filters=64)
bn1 = tf.layers.batch_normalization(conv1)
act1 = tf.nn.relu(bn1)
pool1 = tf.layers.max_pooling2d(inputs = act1,pool_size=(3,3),strides=2)

conv2 = tf.layers.conv2d(inputs=pool1,kernel_size=(5,5),filters=128)
bn2 = tf.layers.batch_normalization(conv2)
act2 = tf.nn.relu(bn2)
pool2 = tf.layers.max_pooling2d(inputs = act2,pool_size=(3,3),strides=2)

conv3 = tf.layers.conv2d(inputs=pool2,kernel_size=(3,3),filters=256)
act3 = tf.nn.relu(conv3)
pool3 = tf.layers.max_pooling2d(inputs = act3,pool_size=(3,3),strides=2)

conv4 = tf.layers.conv2d(inputs=pool3,kernel_size=(3,3),filters=512)
act4 = tf.nn.relu(conv4)
pool4 = tf.layers.max_pooling2d(inputs = act4,pool_size=(2,2),strides=2)

flat = tf.reshape(pool4, [-1,5*5*512])
dense = tf.layers.dense(flat, units=1024, activation=tf.nn.relu)
dr = tf.nn.dropout(dense, keep_prob)
dense1 = tf.layers.dense(dr, units=2048, activation=tf.nn.relu)
dr1 = tf.nn.dropout(dense1, keep_prob)
logits = tf.layers.dense(inputs=dr1, units=classes,activation=tf.nn.sigmoid)#,name='logits')
y_hot = tf.one_hot(indices=Y,depth=classes,dtype = tf.int32)#,name='y_hot')
loss = tf.losses.softmax_cross_entropy(y_hot, logits=logits)
optimizer  = tf.train.AdamOptimizer(1e-3).minimize(loss)

predict = tf.argmax(logits, 1, name='predict')
cp = tf.equal(predict,tf.argmax(y_hot, 1))
accuracy = tf.reduce_mean(tf.cast(cp, tf.float32))


# Training the model


saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Next line is used for further training the model
    #saver.restore(sess,tf.train.latest_checkpoint('./model/'))
    for i in range(epochs):
        losses = []
        for batch_x, batch_y in batch_generator(bs,x_train, y_train):
            batch_x = batch_x.reshape(-1,128,128,1)
            _, lossy = sess.run([optimizer, loss],feed_dict = {X:batch_x,Y :batch_y,keep_prob:0.35})
            losses.append(lossy)
        acc = []
        print('Loss in step {} is - {}'.format(i,np.average(losses)))
        for batch_x, batch_y in batch_generator(bs,x_val, y_val):
            batch_x = batch_x.reshape(-1,128,128,1)
            acc.append(sess.run(accuracy, feed_dict={X:batch_x, Y:batch_y,keep_prob:1.0}))
        print("Accuracy (validation)- ",sum(acc)/float(len(acc)))

        # Saving the model with each five epoch
        if (i+1) % 5 == 0 :
            saver.save(sess,'model/sentiment-{}.ckpt'.format(i))






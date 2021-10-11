
# Lets start dirty and implement a model using tensorflow and try to made it better using several insights.

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np            

# path where your training csv file is being stored
path_train = r'C:\Users\shankul\Desktop\all\sign_data\sign_mnist_train.csv'
# skipping the header 
df = pd.read_csv(path_train,header=None,skiprows=1)

# Visulazing the dataset
print(df.head())

# path where your test csv file is being stored
path_test = r'C:\Users\shankul\Desktop\all\sign_data\sign_mnist_test.csv'

df_test = pd.read_csv(path_test,header=None,skiprows=1)

# converting pandas dataframe to numpy array
x, y = df.loc[:,1:].values, df.loc[:,0].values

x_test, y_test = df_test.loc[:,1:].values, df_test.loc[:,0].values


# Creating classes to label dictionary 

ans = {}
for i in range(65,90):
    ans[i-65] = chr(i) 
del ans[9]



# practically seeing the images with some index
ind = 50000
plt.imshow(x[ind].reshape(28,28),cmap='gray')
print("Label - ",ans[y[ind]])
plt.show()


# importing open cv library
import cv2

# data augmentation tend to provide extra five percent accutracy so followoing is the code for that

# Flipping the images side ways

aug = pd.DataFrame()
y_temp = []
for i in range(x.shape[0]):
    t= cv2.flip(x[i].reshape(28,28),1)
    aug = aug.append(pd.DataFrame(t.reshape(1,28*28)), ignore_index=True)
    y_temp.append(y[i])
    
x = np.vstack((x,aug.values))

y = np.hstack((y,np.array(y_temp)))

# Ramdomly suffling the training samples
from sklearn.utils import shuffle
x, y = shuffle(x, y, random_state=25)

x_train, y_train, x_val, y_val = x[:50000], y[:50000], x[50000:], y[50000:]


# Normalizing the dataset
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# Hyper-parameters 
classes = 24
epochs = 125
bs = 256


# Batch generator
def batch_generator(bs,x,y=None,test=False):
    for i in range(0,len(x),bs):
        if test:
            yield x[i:i+bs]
        else:
            yield x[i:i+bs], y[i:i+bs]


# Input placeholders 
X = tf.placeholder(dtype=tf.float32,name='X',shape=(None,28,28,1))
Y = tf.placeholder(dtype=tf.int32,name='Y',shape=(None))
keep_prob = tf.placeholder(tf.float32,name='keep_prob')


#Network Architecture

conv1 = tf.layers.conv2d(inputs=X,kernel_size=(5,5),filters=64)
act1 = tf.nn.relu(conv1)
pool1 = tf.layers.max_pooling2d(inputs = act1,pool_size=(2,2),strides=2)
conv2 = tf.layers.conv2d(inputs=pool1,kernel_size=(3,3),filters=128)
act2 = tf.nn.relu(conv2)
pool2 = tf.layers.max_pooling2d(inputs = act2,pool_size=(3,3),strides=2)
conv3 = tf.layers.conv2d(inputs=pool2,kernel_size=(3,3),filters=256)
act3 = tf.nn.relu(conv3)
flat = tf.reshape(act3, [-1,2*2*256])
dense = tf.layers.dense(flat, units=516, activation=tf.nn.relu)
dr = tf.nn.dropout(dense, keep_prob)
dense1 = tf.layers.dense(dr, units=1024, activation=tf.nn.relu)
dr1 = tf.nn.dropout(dense1, keep_prob)
logits = tf.layers.dense(inputs=dr1, units=classes,activation=tf.nn.sigmoid)
y_hot = tf.one_hot(indices=Y,depth=classes,dtype = tf.int32)
loss = tf.losses.softmax_cross_entropy(y_hot, logits=logits)
# Adam optimizer
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
            batch_x = batch_x.reshape(-1,28,28,1)
            _, lossy = sess.run([optimizer, loss],feed_dict = {X:batch_x,Y :batch_y,keep_prob:0.35})
            losses.append(lossy)
        acc = []
        print('Loss in step {} is - {}'.format(i,np.average(losses)))
        for batch_x, batch_y in batch_generator(bs,x_val, y_val):
            batch_x = batch_x.reshape(-1,28,28,1)
            acc.append(sess.run(accuracy, feed_dict={X:batch_x, Y:batch_y,keep_prob:1.0}))
        print("Accuracy (validation)- ",sum(acc)/float(len(acc)))

        # Saving the model with each five epoch
        if (i+1) % 5 == 0 :
            saver.save(sess,'model/sign-{}.ckpt'.format(i))


# Measuring training accuracies
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint('./model/'))
    acc=[]
    for batch_x, batch_y in batch_generator(bs,x_train, y_train):
        batch_x = batch_x.reshape(-1,28,28,1)
        acc.append(sess.run(accuracy, feed_dict={X:batch_x, Y:batch_y,keep_prob:1.0}))
    print("Accuracy (train)- ",sum(acc)/float(len(acc))) 


# Measuring test accuracies
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint('./model/'))
    acc=[]
    for batch_x, batch_y in batch_generator(bs,x_test, y_test):
        batch_x = batch_x.reshape(-1,28,28,1)
        acc.append(sess.run(accuracy, feed_dict={X:batch_x, Y:batch_y,keep_prob:1.0}))
    print("Accuracy (train)- ",sum(acc)/float(len(acc)))

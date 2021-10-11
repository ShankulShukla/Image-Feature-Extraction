
# Transfer learning through inception v3 model

import numpy as np
import tensorflow as tf
import os


# model download link - 
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz


# Then extract the tgz file and specify the path below 

path = r"C:\Users\shankul\Desktop\inception-2015-12-05"

# inception class

class inception:
    
    def __init__(self):
        # creating the tensorflow computational graph
        self.graph = tf.Graph()
        # Set the new graph as the default.
        with self.graph.as_default():
            with tf.gfile.FastGFile(os.path.join(path,"classify_image_graph_def.pb"), 'rb') as file:
                # protobuf which can generate specific language stubs, that's where the GraphDef come from
                graph_def = tf.GraphDef()
                # Then we load the proto-buf file into the graph-def.
                graph_def.ParseFromString(file.read())
                # Finally we import the graph-def to the default TensorFlow graph.
                tf.import_graph_def(graph_def, name='')
        
        # if given input with jpeg extension
        self.input_jpeg = self.graph.get_tensor_by_name("DecodeJpeg/contents:0")
        # if given image not jpeg extension
        self.input_not_jpeg = self.graph.get_tensor_by_name("DecodeJpeg:0")
        # unscaled output of softmax classifier
        self.y_pred = self.graph.get_tensor_by_name("softmax/logits:0")
        # used for transfer learning
        self.transfer_layer = self.graph.get_tensor_by_name("pool_3:0")
        # Create a TensorFlow session for executing the graph.
        self.session = tf.Session(graph=self.graph)
   
    def transfer_values(self,x_train):
        # can do this in batches also but as the size of the dataset is small prefered to run single image at a time
        return self.session.run(self.transfer_layer,feed_dict={self.input_not_jpeg:x_train})


import pandas as pd


# Creating labels
l = []
label = []
for folder in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']:
    img_list = os.listdir(os.path.join(r'C:\Users\shankul\Downloads\sign data',folder))
    i = ord(folder)
    temp = [i-65]*len(img_list)
    l.append(len(img_list))
    label.append(temp)

# Converting into numpy array
    
label = np.concatenate(label) 

# Label to classes dictionary
ans = {}
for i in range(65,90):
    ans[i-65] = chr(i)
del ans[9]



def batch_generator(bs,x,y=None,test=False):
    for i in range(0,len(x),bs):
        if test:
            yield x[i:i+bs]
        else:
            yield x[i:i+bs], y[i:i+bs]


# Creating thr object

model = inception()

import cv2


os.chdir(r'C:\Users\shankul\Downloads\sign data')

# Taking each image and finding out the output from the model here we are working with the RGB images 
answer = [] 
for folder in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']:
    img_list = os.listdir(folder)
    print('Loading Folder - {} ...'.format(folder))
    for i,img in enumerate(img_list):
        inp = cv2.imread(os.path.join(folder,img))
        pritn(inp.shape)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp_resize = cv2.resize(inp,None,fx=0.465,fy=0.62)
        answer.append(model.transfer_values(inp_resize))

# Removing single dimension entries from answer and concatenating to form a numpy array

temp = np.squeeze(np.concatenate(answer))

# visualizing some output from the inception model
plt.imshow(temp[0].reshape(32,64))

# applying the PCA to the transfer values of the training images to visualize them in two dimensional space 

from sklearn.decomposition import PCA


pca = PCA(n_components=2)


transfer_values_reduced = pca.fit_transform(temp)


def plot_scatter(values, cls):
    # Create a color-map with a different color for each class.
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, 25))

    # Get the color for each sample.
    colors = cmap[cls]

    # Extract the x- and y-values.
    x = values[:, 0]
    y = values[:, 1]

    # Plot it.
    plt.scatter(x, y, color=colors)
    plt.show()

# Plotting the values 
plot_scatter(transfer_values_reduced, list(ans.keys()))

# dimensionality reduction through t-SNE

from sklearn.manifold import TSNE


pca = PCA(n_components=100)
transfer_values_ = pca.fit_transform(temp)

tsne = TSNE(n_components=2)

transfer_values_reduced = tsne.fit_transform(transfer_values_)

# Plotting the values

plot_scatter(transfer_values_reduced, list(ans.keys()))


# creating extra layer taking input as the transfer values



# placeholder variable for inputting the transfer-values from the Inception model

input_x = tf.placeholder(tf.float32, shape=[None,temp.shape[1]], name="input_x") 
input_y = tf.placeholder(dtype=tf.int32,name='input_y',shape=(None))
keep_prob = tf.placeholder(tf.float32,name='keep_prob')

# Code for printing the tensor plus the opeartion in any protobuf file model
# import tensorflow as tf
# 
# def printTensors(pb_file):
# 
#     # read pb into graph_def
#     with tf.gfile.GFile(pb_file, "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
# 
#     # import graph_def
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def)
# 
#     # print operations
#     for op in graph.get_operations():
#         print(op.name)
# 
# 
# printTensors("classify_image_graph_def.pb")


# Hyperparameters 
classes = len(ans)
epoch = 2000
bs = 128


# creating the layer
fcl1 = tf.layers.dense(input_x,units = 1024,activation = tf.nn.relu)
dr1 = tf.nn.dropout(fcl1, keep_prob)
fcl2 = tf.layers.dense(dr1, units = 512,activation = tf.nn.relu)
dr2 = tf.nn.dropout(fcl2, keep_prob)
logits = tf.layers.dense(inputs=dr2, units=classes, activation=tf.nn.softmax)
y_hot = tf.one_hot(indices=input_y,depth=classes,dtype = tf.int32)



loss = tf.losses.softmax_cross_entropy(y_hot, logits=logits)
optimizer  = tf.train.AdamOptimizer(1e-3).minimize(loss)


predict = tf.argmax(logits, 1, name='predict')
cp = tf.equal(predict,tf.argmax(y_hot, 1))
accuracy = tf.reduce_mean(tf.cast(cp, tf.float32))


# suffling the x and y 
from sklearn.utils import shuffle
x_train, y_train = shuffle(temp, label, random_state=0)


# splitting into training and validation

x_train, y_train, x_val, y_val = x_train[:4500], y_train[:4500], x_train[4500:], y_train[4500:] 

# training the extra layers

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    main_loss = []
    for i in range(epoch):
        losses = []
        for batch_x, batch_y in batch_generator(bs, x_train, y_train):
            _, lossy = sess.run([optimizer, loss],feed_dict = {input_x:batch_x,input_y :batch_y,keep_prob:0.25})
            losses.append(lossy)
        print('Loss in step {} is - {}'.format(i,np.average(losses)))
        main_loss.append(np.average(losses))
        acc = []
        for batch_x, batch_y in batch_generator(bs, x_val,y_val):
            acc.append(sess.run(accuracy, feed_dict ={input_x :batch_x, input_y:batch_y, keep_prob:1.0}))
        print("Accuracy (validation)- ",sum(acc)/float(len(acc)))



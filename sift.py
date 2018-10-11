
# Using sift feature extractor to find the useful features then create a bag of visual words model so fitting a machine learning model on it.


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


# Using dataset created in the sign-language-analysis-improved-dataset python file 


data = pd.read_csv(r'C:\Users\shankul\Downloads\sign data\data.csv')


# Number of samples in each class


le = [242, 259, 247, 147, 243, 226, 241, 116, 179, 245, 182, 235, 237, 229, 234, 216, 211, 229, 216, 135, 122, 128, 202, 251]

# Making labels for the dataset

label = []
for ini,folder in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']):
    i = ord(folder)
    temp = [i-65]*le[ini]
    label.append(temp)



# transforming labels into a numpy array

label = np.concatenate(label) 


# Creating classes to label dictionary 

ans = {}
for i in range(65,90):
    ans[i-65] = chr(i)
del ans[9]


# Giving new names and creating into numpy arrays
x, y = data.values, label


# practically seeing the images with some index
ind = 1090
plt.imshow(x[ind].reshape(-1,128),cmap='gray')
print(ans[y[ind]])
plt.show()

# Visualizing and setting parameters for the edge detector

img = cv2.GaussianBlur(x[87].reshape(-1,128).astype(np.uint8),(5,5),0)
edge = cv2.Canny(img,50,50)
plt.imshow(edge,cmap='gray')
plt.show()


# Applying sift feature extractor on the same image

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(edge,None)
kp, des = sift.compute(edge, kp)
img2 = cv2.drawKeypoints(edge,kp,color=(0,255,0),outImage=None, flags=0)
plt.imshow(img2),plt.show()
print(des.shape[0]) 




def pre_processing(img):
    #as canny edge detector include noise reduction, Non-maximum Suppression,etc we eill use it only
    edge = cv2.Canny(img,60,60)
    return edge



# Moving through each image and extraing the features related to that image

sift = cv2.xfeatures2d.SIFT_create()
temp = []
for i in range(x.shape[0]):
    edge = pre_processing(x[i].reshape(-1,128).astype(np.uint8))
    kp, des = sift.detectAndCompute(edge, None)
    temp.append(des)


# Appending all features (128 element vector) from all images
train_desc = []
for desc_list in temp:
    for desc in desc_list:
        train_desc.append(desc)

# Making it into numpy array

train_desc = np.array(train_desc)

# Seeing for the shape of features found

print(train_desc.shape)


# Lets use k-means clustering to cluster features to create vbag of visual words model

n_clusters = 200
from sklearn.cluster import MiniBatchKMeans
cls = MiniBatchKMeans(n_clusters)


cls.fit(train_desc)


# Now we are going to make histogram of features for each image using cluster centroid found by DBSCAN
pred = []
for desc_list in temp:
    pred.append(cls.predict(desc_list))

# Creating feature vector for the images from the predict cluster of features it is belonging to.
train_hist = [np.bincount(feature,minlength=n_clusters) for feature in pred]


train_hist = np.array(train_hist)


# Fitting an svm model on the feature vector created for each image

from sklearn.svm import SVC 
clf = SVC(kernel='linear')



# randomizing the feature label pair for training 
from sklearn.utils import shuffle
a, b = shuffle(train_hist, label, random_state=25)




#Normalizing the features i.e, divide all columns with its highest value

p = a.max(axis=0)
a_ = []
for i in range(a.shape[0]):
    a_.append(a[i]/(p*1.0))

# creating it into numpy array

a_ = np.array(a_)

# train validation split

x_train, y_train, x_test, y_test = a_[:4400], b[:4400], a_[4400:], b[4400:]

#fitting the model


clf.fit(x_train,y_train)

# train accuracy

print(clf.score(x_train,y_train)*100)


# test accuracy


print(clf.score(x_test,y_test)*100)



from sklearn.ensemble import RandomForestClassifier


clf1 = RandomForestClassifier(n_estimators = 50,n_jobs=-1)



clf1.fit(x_train,y_train)

# train accuracy

print(clf1.score(x_train,y_train)*100)

# test accuracy


print(clf1.score(x_test,y_test)*100)



from sklearn.neural_network import MLPClassifier
clf3 = MLPClassifier()

clf3.fit(x_train,y_train)

# train accuracy

print(clf1.score(x_train,y_train)*100)

# test accuracy


print(clf1.score(x_test,y_test)*100)



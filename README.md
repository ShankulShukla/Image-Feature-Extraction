# Analysis on image feature extraction

An implementational study on traditional and deep learning techniques for feature extraction. Extracting features i.e., comprehensive description of the image to enhance the accuracy of our classifier on the target dataset. This project involves various image processing techniques including *edge detection, data augmentation, smoothing, feature detection, and extraction, etc*. 


## Traditional feature detection

In this, we extract a set of descriptors of the image’s features, then pass those extracted features to our machine learning algorithms for classification on Hand sign language classification. Before extracting features from feature detection algorithms we apply some processing steps to our images - 

Images in our dataset contain different types of noise due to their creation via webcam, image smoothing (also called blurring) helps in reducing the noise. We applied a Gaussian filter for image smoothing to all images in our dataset as we found that gaussian preserves more features like edges which is quite important based on our hand sign dataset. The Gaussian kernel size of 5x5 gave the best results. Then I extracted the edges using a canny edge detector and passed these processed images to the feature extraction algorithms explained in this section. I observed that this processing resulted in an increased set of feature descriptors for each algorithm and also better accuracy on the target dataset.

Further, I used the bag of visual word models on the feature extracted by the descriptors and applied K means clustering so that similar features are clustered together. Then created a new feature representation of the images based on the clustering.

> Based on the feature representation created, I tried various Machine learning algorithms and got the following results (test accuracy) on the improved sign dataset-

![image](images/accuracytrad.png)

> With the Sign MNIST dataset all algorithms were able to achieve more than 90% test accuracy.

### Scale-Invariant Feature Transform (SIFT) 

- SIFT proposed by Lowe solves the image rotation, affine
transformations, intensity, and viewpoint change in matching
features. 
- The SIFT algorithm has 4 basic steps- 
  - First is to estimate scale-space extrema using the Difference of
Gaussian (DoG). 
  - Secondly, a key point localization where the
key point candidates are localized and refined by eliminating
the low contrast points. 
  - Thirdly, a key point orientation
assignment based on local image gradient
  - Lastly a
descriptor generator to compute the local image descriptor for
each key point based on image gradient magnitude and
orientation

- The SIFT detector is rotation-invariant and scale-invariant.
- Although SIFT has
proven to be very efficient in object recognition applications,
it requires a large computational complexity which is a major
drawback especially for real-time applications

> Visualization of feature extracted from SIFT - 

![image](images/sift1.png)
![image](images/sift2.png)

[Main Paper](https://people.eecs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf)

### Speeded-Up Robust Features (SURF)

- Speed up Robust Feature (SURF) technique, which is an
approximation of SIFT performs faster than SIFT without
reducing the quality of the detected points.

- SURF approximates the DoG with box filters. Instead of
Gaussian averaging the image, squares are used for
approximation since the convolution with square is much
faster if the integral image is used. Also, this can be done in
parallel for different scales. 

- The SURF uses a BLOB detector
which is based on the Hessian matrix to find the points of
interest. For orientation assignment, it uses wavelet responses
in both horizontal and vertical directions by applying adequate
Gaussian weights. For feature description also SURF uses the
wavelet responses.

> Visualization of feature extracted from SURF - 

![image](images/surf1.png)
![image](images/surf2.png)

[Main Paper](https://people.ee.ethz.ch/~surf/eccv06.pdf)

### Oriented FAST and Rotated BRIEF (ORB)

- ORB is a fusion of the FAST keypoint detector and BRIEF
descriptor with some modifications. 
- Initially, to determine
the key points, it uses FAST. Then a Harris corner measure is
applied to find top N points. FAST does not compute the
orientation and is a rotation variant. It computes the intensity
weighted centroid of the patch with the located corner at the center.
- The direction of the vector from this corner point to centroid
gives the orientation. Moments are computed to improve the
rotation invariance. The descriptor BRIEF poorly performs if
there is an in-plane rotation. 
- In ORB, a rotation matrix is
computed using the orientation of patch and then the BRIEF
descriptors are steered according to the orientation. 

- SIFT and SURF are patented and this algorithm from OpenCV labs is a free alternative to them.


> Visualization of feature extracted from ORB - 

![image](images/orb1.png)
![image](images/orb2.png)

[Main Paper](https://ieeexplore.ieee.org/document/6126544)

## Deep Learning feature extraction

Traditional feature extractors can be replaced by a convolutional neural network(CNN), since CNN’s have a strong ability to extract complex features that express the image in much more detail, learn the task-specific features, and are much more efficient.

### CNN model with batch-normalization

- Created a four-layered CNN-based model with max-pooling and three fully connected layers.
- Batch normalization is applied in the first two layers as it tends to perform better results.
- It also consists of dropout, ReLU activations, ADAM optimizer with softmax cross-entropy cost function.
> With 50 epochs of training, the model achieved **97%** accuracy on the test set of the Sign MNIST dataset.

> With 100 epochs of training, the model achieved **94.5%** accuracy on the test set of the improved dataset.

> With 125 epochs of training, the model got an accuracy of **98.28%** on the test set of the improved dataset.

> Visualization of Feature Maps learned in each layer from the dataset- 

![image](images/cnnfeatureout.png)
![image](images/cnnfeatureout2.png)
![image](images/cnnfeatureout3.png)
![image](images/cnnfeatureout4.png)
![image](images/cnnfeatureout5.png)

[AlexNet based Feature extraction paper](https://www.semanticscholar.org/paper/Feature-extraction-and-image-retrieval-based-on-Yuan-Zhang/bada07c7ea423739c0db6b8f1f2fc2438881f21d)

### Transfer learning with Inception v3 model
- Deep neural networks trained on natural images learn similar features (texture, corners, edges, and color blobs) in the initial layers. Such initial-layer features appear not to be specific to a particular data-set or task but are general in that they are applicable to many datasets and tasks. We call these initial-layer features general and can be transferred for learning specific data-set.
- Pre-trained Inception model is being used and an extra layer is being added to create a new inception architecture to classify hand signs.
- To decrease the compute time I stored the transfer values by the model of the training samples ahead of training and saved them onto a pickle file.
- With 500 epochs of training our model tend to achieve a test accuracy of **95%** on the improved dataset and this model also tends to perform well on the webcam input on testing.

> Webcam application detecting this sign as “s” - 

![image](images/pred.png)

[Application of transfer learning in feature extraction paper](https://ieeexplore.ieee.org/document/7946733)

## Dataset

### Sign MNIST based model
- I started with a very basic dataset available at this [link](https://www.kaggle.com/datamunge/sign-language-mnist) 
- The dataset has 27455 training samples which are 28 * 28 pixels so when this model is tested on webcam input it does not tend to perform well given getting very good accuracy in the test set.

### Improved sign dataset
- [This](https://drive.google.com/open?id=1wgXtF6QHKBuXRx3qxuf-o6aOmN87t8G-) dataset is hand created is much more difficult to work with having 480 * 640 images and reflects good properties to make the model deployable for real-world hand inputs.
- This dataset is created by webcam from scratch for each static American hand sign language.
- The images in the dataset are resized to 128 * 128 pixels for our use.


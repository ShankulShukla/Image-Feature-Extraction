Hand-Sign-Detection
Implemented various techniques for static hand sign understanding. Initially started out with the MNIST sign dataset available in the kaggle and then moving on a more bigger dataset available at this [link](https://drive.google.com/open?id=1wgXtF6QHKBuXRx3qxuf-o6aOmN87t8G-). I applied various image processing technique including edge detection, feature detection, and extraction etc. Also, I used CNN architecture as well as transfer learning using inception model, I also used various machine learning classifier which works on image feature extracted.

- Give comprehensive description of image.

## Traditional feature detection

In this we extract set of descriptors of image’s features, then pass those extracted features to our machine learning algorithms for classification on hand sign language classification. Before extracting features from feature detection algorithms we applied two processing steps to our images - 

Images can contain different types of noise, image Smoothing (also called blurring) techniques help in reducing the noise. We applied Gaussian filter for image smoothing to all images in our dataset as we found that gaussian preserves more features like edges which is quite important based on our hand sign dataset. The kernel size of 5x5 gave the best results. Then I extracted the edges using canny edge detector and passed these processed imahge to the feature extraction alorgtims defined later. This earlier processing resulted in increased set of feature descriptor for each algorihtm and also better accuracy on the target dataset.

### Scale-Invariant Feature Transform (SIFT) 

- SIFT proposed by Lowe solves the image rotation, affine
transformations, intensity, and viewpoint change in matching
features. 
- The SIFT algorithm has 4 basic steps- 
  - First is to estimate a scale space extrema using the Difference of
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

[Main Paper](https://people.eecs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf)

### Speeded-Up Robust Features (SURF)

- Speed up Robust Feature (SURF) technique, which is an
approximation of SIFT, performs faster than SIFT without
reducing the quality of the detected points.

- SURF approximates the DoG with box filters. Instead of
Gaussian averaging the image, squares are used for
approximation since the convolution with square is much
faster if the integral image is used. Also this can be done in
parallel for different scales. 

- The SURF uses a BLOB detector
which is based on the Hessian matrix to find the points of
interest. For orientation assignment, it uses wavelet responses
in both horizontal and vertical directions by applying adequate
Gaussian weights. For feature description also SURF uses the
wavelet responses

[Main Paper](https://people.ee.ethz.ch/~surf/eccv06.pdf)

### Oriented FAST and Rotated BRIEF (ORB)

- ORB is a fusion of the FAST key point detector and BRIEF
descriptor with some modifications. 
- Initially to determine
the key points, it uses FAST. Then a Harris corner measure is
applied to find top N points. FAST does not compute the
orientation and is rotation variant. It computes the intensity
weighted centroid of the patch with located corner at center.
- The direction of the vector from this corner point to centroid
gives the orientation. Moments are computed to improve the
rotation invariance. The descriptor BRIEF poorly performs if
there is an in-plane rotation. 
- In ORB, a rotation matrix is
computed using the orientation of patch and then the BRIEF
descriptors are steered according to the orientation. 

- SIFT and SURF are patented and this algorithm from OpenCV labs is a free alternative to them, that uses FAST keypoint detector and BRIEF descriptor.

[Main Paper](https://ieeexplore.ieee.org/document/6126544)

## Deep Learning feature extraction

### AlexNet

[AlexNet based Feature extraction paper](https://www.semanticscholar.org/paper/Feature-extraction-and-image-retrieval-based-on-Yuan-Zhang/bada07c7ea423739c0db6b8f1f2fc2438881f21d)
### Transfer learning with Inception 

[Application of transfer learning in feature extraction paper](https://ieeexplore.ieee.org/document/7946733)


## Sign MNIST based model
- I started with very basic dataset available at this [link](https://www.kaggle.com/datamunge/sign-language-mnist) and created a three-layered CNN based model with max-pooling at first two layers and two fully connected layers.
- With 125 epochs of training, the model got an accuracy of **98.28%** on the test set.
- The dataset has 27455 training samples which are 28 * 28 pixels so when this model tested on webcam input it does not tend to perform well given getting very good accuracy in the test set.

## Improved sign dataset
- [This](https://drive.google.com/open?id=1wgXtF6QHKBuXRx3qxuf-o6aOmN87t8G-) dataset is hand created is much more difficult to work with having 480 * 640 images and reflects good properties to make model deployable for real-world hand inputs.
- Similar model architecture is being used as for sign MNIST but an extra convolution layer is added with max-pooling layer. Batch normalization is applied in the first two layers as it tends to perform better results.
- WIth 100 epochs of training, the model achieved **94.5%** accuracy on the test set.
> #Note- The images in the dataset is resized to 128 * 128 pixels and also to grayscale.

## Image classification using feature extractors
- Here I tend to work on classic image processing techniques for hand sign recognition.
- I used the improved dataset and applied edge detection and then try to extract the features in the image by suing feature extractor algorithms like SIFT, SURF, and ORB. Further, I used the bag of visual word model on the feature extracted by the descriptors and applied K means clustering so that similar features are clustered together. Then created a new feature representation of the images based on the clustering.
- On the new feature representation found on the previous step I tried various Machine learning model and got the following results.
![](image.png)


## Transfer learning using inception v3 model
- Pre-trained Inception model is being used and an extra layer is being added onto to create a new inception architecture to classify hand sign.
- To decrease the compute time I stored the transfer values by the model of the training samples ahead of training and saved it onto a pickle file.
- With 500 epochs of training our model tend to achieve an accuracy of **95%** and this model also tend to perform well on the webcam input as the images with which the model is trained is RGB images not the gray-scale.

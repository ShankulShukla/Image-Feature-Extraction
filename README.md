# Hand-Sign-Detection
Implemented various techniques for static hand sign understanding. Initially started out with the MNIST sign dataset available in the kaggle and then moving on a more bigger dataset available at this [link](https://drive.google.com/open?id=1wgXtF6QHKBuXRx3qxuf-o6aOmN87t8G-). I applied various image processing technique including edge detection, feature detection, and extraction etc. Also, I used CNN architecture as well as transfer learning using inception model, I also used various machine learning classifier which works on image feature extracted.

## Sign MNIST based model
- I started with very basic dataset available at this [link](https://www.kaggle.com/datamunge/sign-language-mnist) and created a three layered CNN based model with max-pooling at first two layers and two fully conected layers.
- With 125 epochs of training the model got accuracy of 98.28% on test set.
- The dataset have 27455 training samples wich are 28 * 28 pixel so when this model tested on webcam input it does not tend to perform well given getting very good accuracy in test set.

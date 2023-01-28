# MNIST-classification-using-CNN

The repository contains the data & code for classifying the Images of handwritten digits using Convolutional Neural Network(CNN) and Dropouts approach.

## Dataset

Loading the MNIST data set directly using TensorFlow API. 

```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

First we check the distribution of classes in the training set.


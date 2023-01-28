# MNIST-classification-using-CNN

The repository contains the data & code for classifying the Images of handwritten digits using the Convolutional Neural Network(CNN) and Dropouts approach.

## Dataset

Loading the MNIST data set directly using TensorFlow API. 

```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

First, we check the distribution of classes in the training set.

![data_distribution](https://github.com/Mayurwaghela1997/MNIST-classification-using-CNN/blob/main/Images/Class_distribution.JPG)

We can see there is no class imbalance problem in the training set.

Next, Let's have a look at a random image and label for that image from the training set.


![Sample_data](https://github.com/Mayurwaghela1997/MNIST-classification-using-CNN/blob/main/Images/Sample_data.JPG)


## Normalizing & Reshaping the Images

The TensorFlow Conv2D layer expects the data in 4 dimensions (w,x,y,z) tensor, but the current dimension for of data is 3.

```python
x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
```

Once the data is reshaped, we must normalize it to have data on a standard scale of 0 to 1.
We divide the data with 255 because the RGB values of the image range from 0 to 255.

```python
x_train / 255.0
```

## One-Hot Encoding of the Labels
The label ranges from 0 to 9. In order to make a classification, the model expects the data in the form of encodings. 
We use TensorFlow one-hot encoding method to encode the Labels.

For e.g. 
```python
Label_before_encoding = 8
Label_after_encoding = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
```

## Model Summary & Training

The below image represents the architecture of the sequential model. 

![model](https://github.com/Mayurwaghela1997/MNIST-classification-using-CNN/blob/main/Images/model_summary.JPG)

Here, we use Max pooling to reduce the computation and increase the training speed.
We also use Dropout, where we randomly nullify neurons to avoid overfitting of data.

For training the model, we use an 80/20 split of training data and use EarlyStopping callback to stop training when loss doesn't change for 3 Epochs.

The below Image represents the training history of the model.

![history](https://github.com/Mayurwaghela1997/MNIST-classification-using-CNN/blob/main/Images/Training_history.JPG)

## Evaluating the model

The model was evaluated on unseen data and achieved 98.94% Accuracy.

The below image represents the confusion matrix of the evaluation.

![confusion_matrix](https://github.com/Mayurwaghela1997/MNIST-classification-using-CNN/blob/main/Images/Confusion_matrix.JPG)

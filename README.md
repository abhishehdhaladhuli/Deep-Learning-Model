
# Introduction

This project focuses on building a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset contains 60,000 32x32 color images divided into 10 classes, with 6,000 images per class. The primary goal is to develop a model that can accurately categorize these images into their respective classes, such as airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

The project involves several essential steps:

- Importing necessary libraries
- Loading and preprocessing the dataset
- Designing and compiling a CNN model
- Training the model and visualizing training metrics
- Evaluating the model's performance on test data
- Displaying a confusion matrix to understand classification results

The following is a detailed breakdown of the entire code.

## TensorFlow and Keras Import
The code begins by importing essential libraries:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
```

- `tensorflow` and `keras` are used for building and training the neural network.
- `matplotlib.pyplot` is used to plot graphs for training history and the confusion matrix.
- `numpy` is used for array manipulations.
- `confusion_matrix` and `ConfusionMatrixDisplay` from `sklearn.metrics` help evaluate the classification results.

## Dataset Loading and Preprocessing

The CIFAR-10 dataset is loaded and preprocessed:

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
```

- `keras.datasets.cifar10.load_data()` loads the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes.
- Pixel values range from 0 to 255, so they are normalized to the range [0, 1] by dividing by 255.0.

## CNN Model Definition

A convolutional neural network (CNN) is defined using Keras' Sequential API:

```python
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

Explanation of layers:
- `layers.Input(shape=(32, 32, 3))`: Defines the input layer for 32x32 RGB images.
- `Conv2D(32, (3, 3), activation='relu')`: A convolutional layer with 32 filters and 3x3 kernels using ReLU activation.
- `MaxPooling2D((2, 2))`: A pooling layer that reduces spatial dimensions by 2x2.
- Another pair of Conv2D and MaxPooling2D layers with increased filters (64).
- `Flatten()`: Flattens the 2D feature maps into a 1D vector.
- `Dense(64, activation='relu')`: A fully connected layer with 64 units and ReLU activation.
- `Dense(10, activation='softmax')`: Output layer with 10 units (one per class) using softmax activation.

## Model Compilation

The model is compiled with the Adam optimizer, sparse categorical crossentropy loss, and accuracy metric:

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

- `adam`: An adaptive learning rate optimization algorithm.
- `sparse_categorical_crossentropy`: Used because labels are integers (not one-hot encoded).
- `accuracy`: To track accuracy during training and validation.

## Model Training

The model is trained for 10 epochs with training and validation data:

```python
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)
```

- `x_train` and `y_train`: Training data and labels.
- `epochs=10`: The number of passes through the entire dataset.
- `validation_data=(x_test, y_test)`: Evaluates model performance on test data after each epoch.
- `verbose=1`: Displays progress bar during training.

## Visualizing Training History

The training history (accuracy and loss) is plotted using Matplotlib:

```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

- The first subplot shows training and validation accuracy over epochs.
- The second subplot shows training and validation loss over epochs.
- `plt.show()` renders the plots.

## Model Evaluation

The trained model is evaluated on the test set:

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy:.2f}')
```

- `model.evaluate()` calculates loss and accuracy for test data.
- The test accuracy is printed.

## Confusion Matrix

Finally, a confusion matrix is generated and visualized:

```python
y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
```

- `model.predict(x_test)` generates predictions for test images.
- `np.argmax()` gets class labels with highest probability.
- `confusion_matrix()` computes the confusion matrix.
- `ConfusionMatrixDisplay` visualizes it with a blue colormap.

## Summary

This CNN model classifies images from CIFAR-10, training through 10 epochs and visualizing accuracy and loss. The model is then evaluated using a confusion matrix, providing a clear understanding of its classification performance.



# Handwritten_digit_recognizer
Python script that implements a handwritten digit recognizer using a convolutional neural network (CNN) model. It accomplishes several tasks, including loading and preprocessing data, building and training the neural network, evaluating its performance, and using it to predict digits in an input image.

## Overview

This repository contains a Python script that demonstrates the implementation of a handwritten digit recognizer using a convolutional neural network (CNN) model. The script performs the following key tasks:

- Importing necessary libraries for data manipulation, machine learning, and image processing.
- Defining the `HandwrittenDigitRecognizer` class that encapsulates the entire digit recognition process.
- Loading and preprocessing the MNIST dataset, preparing it for training and testing.
- Building a CNN model for digit recognition using Keras.
- Training the model on the MNIST dataset.
- Evaluating the trained model's performance on test data.
- Predicting digits in input images using the trained model.

## Code Explanation

The script follows these main steps:

1. **Importing Libraries:** The script begins by importing essential libraries including NumPy, TensorFlow, Keras, and PIL (Python Imaging Library) for image handling.

2. **HandwrittenDigitRecognizer Class:** The `HandwrittenDigitRecognizer` class is defined to manage the entire digit recognition process. It includes methods to load data, build a CNN model, train the model, evaluate its performance, and predict digits in an image.

3. **Loading Data:** The `load_data` method loads and preprocesses the MNIST dataset. It reshapes and normalizes the images, and one-hot encodes the labels.

4. **Building the Model:** The `build_model` method constructs the CNN model using Keras. The model includes convolutional and pooling layers, followed by fully connected layers for classification.

5. **Compiling the Model:** The `compile` method configures the model for training, specifying the optimizer, loss function, and evaluation metric.

6. **Training the Model:** The `train_model` method trains the CNN model on the training data using the `fit` function.

7. **Evaluating the Model:** The `evaluate_model` method assesses the trained model's performance on the test data by computing accuracy and loss.

8. **Predicting Digits in an Image:** The `predict_digits_in_image` method loads an input image, preprocesses it, and predicts the digits present using the trained model.

## Usage

1. Ensure you have the necessary libraries installed (`numpy`, `tensorflow`, `keras`, and `PIL`).
2. Replace `'7.jpeg'` in the code with the actual path to the image you want to test.
3. Run the script.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to explore and modify the script for your own purposes.

For more information on the MNIST dataset and Keras library, refer to the official documentation.


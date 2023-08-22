# Import necessary libraries
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from PIL import Image


# Define a class for the handwritten digit recognizer
class HandwrittenDigitRecognizer:
    def __init__(self):
        self.model = None  # Placeholder for the neural network model
        self.optimizer = tf.keras.optimizers.legacy.Adadelta(learning_rate=1.0)  # Optimizer for the model

    def load_data(self):
        # Load the MNIST dataset and preprocess it
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.0
        self.x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.0
        self.y_train = to_categorical(y_train, num_classes=10)
        self.y_test = to_categorical(y_test, num_classes=10)

    def build_model(self):
        # Build a neural network model for digit recognition
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))
        
        # Compile the model with optimizer and loss function
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs=2, batch_size=64):
        # Train the neural network model using training data
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.x_test, self.y_test))

    def evaluate_model(self):
        # Evaluate the trained model's performance on test data
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print("Test accuracy:", test_acc)
        print("Test loss:", test_loss)

    def predict_digits_in_image(self, image_path):
        # Load an image, preprocess it, and predict the digits it contains
        image = Image.open(image_path).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image).astype('float32') / 255
        image_array = image_array.reshape(-1, 28, 28, 1)
        predicted_probabilities = self.model.predict(image_array)
        predicted_digits = np.argmax(predicted_probabilities, axis=1)
        print("Predicted digits:", predicted_digits)

if __name__ == "__main__":
    # Main execution block
    recognizer = HandwrittenDigitRecognizer()  # Create an instance of the digit recognizer
    recognizer.load_data()  # Load and preprocess the dataset
    recognizer.build_model()  # Build the neural network model
    recognizer.train_model()  # Train the model using the dataset
    recognizer.evaluate_model()  # Evaluate the model's performance
    
    image_path = '7.jpeg'  # Replace with the actual path to the image
    recognizer.predict_digits_in_image(image_path)  # Predict digits in the specified image

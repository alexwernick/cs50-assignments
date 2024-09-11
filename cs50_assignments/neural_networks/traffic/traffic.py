import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

EPOCHS = 100
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
MODEL_NUMBER = 1
APPLY_FIXED_KERNAL_TO_IMAGE = False


def main():
    # Check command-line arguments
    # if len(sys.argv) not in [2, 3]:
    #      sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    dir = os.path.join(
        Path(__file__).parent, "gtsrb" if len(sys.argv) < 2 else sys.argv[1]
    )
    images, labels = load_data(dir)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    model.summary()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images, labels = [], []

    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)

    for category in range(NUM_CATEGORIES):
        for image_file in os.listdir(os.path.join(data_dir, str(category))):
            image_path = os.path.join(data_dir, str(category), image_file)
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Image at path {image_path} could not be loaded.")

            resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            if APPLY_FIXED_KERNAL_TO_IMAGE:
                resized_image = cv2.filter2D(resized_image, -1, kernel)

            np_array = np.array(resized_image)
            images.append(np_array)
            labels.append(category)
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    if MODEL_NUMBER == 1:
        # convolutional base
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            )
        )
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))

        # flatten the output before feeding into dense layers
        model.add(tf.keras.layers.Flatten())

        # dense layers
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(NUM_CATEGORIES))

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model
    elif MODEL_NUMBER == 2:
        # convolutional base
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            )
        )
        model.add(tf.keras.layers.Dropout(0.8))  # Dropout applied to the input layer
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))

        # flatten the output before feeding into dense layers
        model.add(tf.keras.layers.Flatten())

        # add a Dropout layer
        model.add(
            tf.keras.layers.Dropout(0.5)
        )  # 50% of the neurons will be dropped during training

        # dense layers
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(NUM_CATEGORIES))

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model
    else:
        return None


if __name__ == "__main__":
    main()

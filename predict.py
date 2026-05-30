import os
import sys

import numpy as np
import tensorflow as tf


IMAGE_SIZE = (150, 150)
MODEL_PATH = os.path.join("model", "cat_dog_classifier.keras")

def prepare_image(image_path):
    "load an image file and turn it into a batch the model can read."
    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image)

    # the model expects a batch of images, so add one extra batch dimension.
    image_batch = np.expand_dims(image_array, axis=0)

    return image_batch

"print Cat or Dog with the prediction confidence percentage."
def predict_image(image_path):
    if not os.path.isfile(MODEL_PATH):
        print(f"Trained model not found at {MODEL_PATH}")
        print("Train the model first by running: python main.py")
        return

    if not os.path.isfile(image_path):
        print(f"Image not found: {image_path}")
        return
    
    model = tf.keras.models.load_model(MODEL_PATH)
    image_batch = prepare_image(image_path)

    # The sigmoid output is a number from 0 to 1.
    # Values closer to 1 mean Dog, and values closer to 0 mean Cat.
    prediction_value = model.predict(image_batch, verbose=0)[0][0]

    if prediction_value >= 0.5:
        label = "Dog"
        confidence = prediction_value * 100
    else:
        label = "Cat"
        confidence = (1 - prediction_value) * 100

    print(f"Prediction: {label} ({confidence:.1f}%)")

"read the image path from the command line and run prediction."
def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        return

    predict_image(sys.argv[1])


if __name__ == "__main__":
    main()
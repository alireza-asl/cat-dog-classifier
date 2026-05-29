import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# image settings are grouped near the top so beginners can customize them easily.
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 5


# folder paths is used by the training script.
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALIDATION_DIR = os.path.join(DATASET_DIR, "validation")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "cat_dog_classifier.keras")

def load_datasets():
    """Load training and validation images from folders."""
    # image_dataset_from_directory automatically labels images from folder names.
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )



# prefetching prepares the next batch while the current batch is training.
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, validation_dataset


def build_model():
    model = models.Sequential(
        [
            # normalize pixel values from 0-255 to 0-1.
            layers.Rescaling(1.0 / 255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),

            # first convolution block: find simple patterns such as edges.
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            # second convolution block: combine simple patterns into shapes.
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            # third convolution block: learn higher-level image features.
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            # flatten converts image features into one long list for Dense layers.
            layers.Flatten(),

            # dropout randomly ignores some neurons during training to prevent and reduce overfitting.
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),

            # sigmoid outputs: close to 0 = Cat, close to 1 = Dog.
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model



def plot_training_history(history):
    """Show accuracy and loss charts for training and validation data."""
    accuracy = history.history["accuracy"]
    validation_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    validation_loss = history.history["val_loss"]
    epochs_range = range(1, len(accuracy) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label="Training Accuracy")
    plt.plot(epochs_range, validation_accuracy, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, validation_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()



def main():
    """Train the CNN, show charts, and save the trained model."""
    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VALIDATION_DIR):
        print("Dataset folders were not found.")
        print("Please create this structure before training:")
        print("dataset/train/cats, dataset/train/dogs")
        print("dataset/validation/cats, dataset/validation/dogs")
        return
    

    train_dataset, validation_dataset = load_datasets()
    model = build_model()

    model.summary()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
    )

    plot_training_history(history)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
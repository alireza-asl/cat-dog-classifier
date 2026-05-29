This project is intended for educational purposes and uses a small dataset to demonstrate image classification with CNNs and TensorFlow. During experimentation, the model was trained with 5, 10, and 15 epochs. While increasing the number of epochs improved training accuracy, it also increased the risk of overfitting due to the limited dataset size. As a result, 5 epochs provided more stable validation performance and were selected for this project. The impact of different epoch values can be observed through the generated training and validation accuracy/loss graphs included in the project.


# project Structure

```text
cat and dog classifier/
│
├── dataset/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── validation/
│       ├── cats/
│       └── dogs/
├── model/
├── images/
├── main.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore

# features

 uses TensorFlow/Keras
 builds a CNN from scratch
 normalizes image data from `0-255` to `0-1`
 uses ReLU activation in hidden layers
 uses sigmoid activation in the output layer
 adds dropout layers to reduce overfitting
 shows training and validation accuracy/loss graphs with matplotlib
 saves the trained model after training
 predicts custom images with confidence percentage
 includes comments to help beginners learn from the code

 # how the CNN Works

a **Convolutional Neural Network (CNN)** is commonly used for image classification because it can learn visual patterns from images.

In this project:

1. **rescaling** normalizes image pixels from `0-255` to `0-1`.
2. **conv2D layers** learn patterns such as edges, textures, ears, eyes, and fur.
3. **maxPooling2D layers** reduce image size while keeping important features.
4. **flatten** converts learned image features into a list of numbers.
5. **dropout** helps reduce overfitting by randomly ignoring some neurons during training.
6. **dense layers** make the final decision.
7. **sigmoid output** returns one probability:
    closer to `0` means **Cat**
    closer to `1` means **Dog**


    #start:
    # 1- Create and activate a virtual environment
    python -m venv .venv
    .venv\Scripts\Activate.ps1

    # 2-Install requirements.
    pip install -r requirements.txt

    Run:
    python main.py

    # Beginner Notes

     more training images usually improve results.
     validation accuracy helps show how well the model performs on images it has not trained on.
     if training accuracy is high but validation accuracy is low, the model may be overfitting.
     dropout helps reduce overfitting, but a better dataset is often the biggest improvement.


during training, the script will:

1. Load cat and dog images from the dataset folders
2. Train the CNN model
3. Display training and validation accuracy/loss graphs
4. Save the model to:
model/cat_dog_classifier.keras
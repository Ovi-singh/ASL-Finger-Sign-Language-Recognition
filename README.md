# ASL-Finger-Sign-Language-Recognition

This project is a machine learning model that recognizes hand gestures representing American Sign Language (ASL) letters. It uses a Convolutional Neural Network (CNN) to classify images of hand gestures into one of 7 classes.

How It Works:

Dataset: The dataset should consist of 128x128 pixel images. Each directory in the dataset should be named after the corresponding ASL letter (e.g., "A", "B", etc.).
Training: Run the train.py script to train the model. The model will be saved as asl_model.h5. If you want to increase the dataset, simply add more images to the appropriate directories and retrain the model.
Prediction: After training, use the main.py script to load the model and make predictions on uploaded images.


Instructions:

Prepare your dataset with 128x128 images, organized into directories named by letter (e.g., "A", "B", etc.).
Run the train.py script to train the model. It will create or update the asl_model.h5 file.
Run the main.py script to use the trained model for predictions.


Future Improvements:
Add more ASL signs to expand the dataset.
Optimize the model for better accuracy.

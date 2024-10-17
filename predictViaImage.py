import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('handwritten_digit_model.keras')

# Load and preprocess the image
# Load and preprocess the image
image = cv2.imread('example1.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))  # Resize to match model input
image = image / 255.0  # Normalize to [0, 1]
image = image.reshape(1, 28, 28, 1)  # Reshape for model input


# Make a prediction
predictions = model.predict(image)
predicted_digit = np.argmax(predictions)
print(f'Predicted Digit: {predicted_digit}')

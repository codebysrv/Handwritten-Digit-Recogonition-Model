import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('handwritten_digit_model.keras')

# Load the image in grayscale mode
image = cv2.imread('example_digit_1.png', cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Unable to open or find the image at {'example1.png'}")
else:
    # Resize the image to 28x28 pixels as expected by the model
    image = cv2.resize(image, (28, 28))
    
    # Normalize the image to the range [0, 1]
    image = image / 255.0
    
    # Reshape the image to (1, 28, 28, 1) to match the model's expected input shape
    image = image.reshape(1, 28, 28, 1)
    
    # Make a prediction
    predictions = model.predict(image)
    
    # Get the digit with the highest probability
    predicted_digit = np.argmax(predictions)
    
    print(f'Predicted Digit: {predicted_digit}')


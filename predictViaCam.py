import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('handwritten_digit_model.keras')

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the frame to match model input
    resized = cv2.resize(gray, (28, 28))  # Resize to 28x28
    normalized = resized / 255.0  # Normalize to [0, 1]
    reshaped = normalized.reshape(1, 28, 28, 1)  # Reshape for model input

    # Make a prediction
    predictions = model.predict(reshaped)
    predicted_digit = np.argmax(predictions)

    # Display the prediction on the frame
    cv2.putText(frame, f'Predicted Digit: {predicted_digit}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Handwritten Digit Recognition', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

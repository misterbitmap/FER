import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("emotion_recognition_model.h5")



def check_emotion(image_path):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0
    
    # Map the prediction to the corresponding emotion
    prediction = model.predict(img)
    emotion_index = np.argmax(prediction)
    predicted_emotion = emotions[emotion_index]
    
    print(f"Predicted emotion: {predicted_emotion}")
    print(f"Confidence: {prediction[0][emotion_index]:.2f}")

# Usage
# check_emotion("dataset/test/happy/PublicTest_99626406.jpg")
check_emotion("test/02.jpg")
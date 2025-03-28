import cv2
import numpy as np
from tensorflow.keras.models import load_model

class FacialExpressionRecognizer:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_face(self, frame):
        """Detect face in a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return faces

    def preprocess_image(self, img):
        """Preprocess image for model input."""
        img = cv2.resize(img, (48, 48))
        img = img / 255.0
        img = img.reshape(1, 48, 48, 1)
        return img

    def predict_emotion(self, img):
        """Predict emotion using the trained model."""
        prediction = self.model.predict(img)
        emotion_index = np.argmax(prediction)
        return self.emotions[emotion_index]

    def recognize_expression(self):
        """Capture video and recognize facial expressions in real-time."""
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame,1)
            
            faces = self.detect_face(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                roi_gray = self.preprocess_image(roi_gray)
                emotion = self.predict_emotion(roi_gray)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 2)

            cv2.imshow('Facial Expression Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    recognizer = FacialExpressionRecognizer("emotion_recognition_model.h5")
    recognizer.recognize_expression()

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load pre-trained model
@st.cache_resource
def load_emotion_model():
    return load_model('emotion_model.h5')

model = load_emotion_model()

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Preprocessing for uploaded images
def preprocess_face(img):
    face_resized = cv2.resize(img, (48, 48))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_norm = face_rgb / 255.0
    face_input = np.reshape(face_norm, (1, 48, 48, 3))
    return face_input

# Title
st.title("Ashwita's Player Emotion Classifier 🎯")

# Upload up to 10 images
uploaded_files = st.file_uploader("📤 Upload up to 10 player images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.warning("⚠️ You can upload a maximum of 10 images at a time.")
    else:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        for idx, uploaded_file in enumerate(uploaded_files):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)

            # Display original image
            st.image(img, caption=f"Player {idx+1}", use_column_width=True)

            # Face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(faces) == 0:
                st.warning(f"❌ No face detected in Player {idx+1}'s image.")
            else:
                # Take the first face found
                (x, y, w, h) = faces[0]
                face_img = img[y:y + h, x:x + w]

                # Preprocess and predict
                input_face = preprocess_face(face_img)
                preds = model.predict(input_face)[0]
                emotion_index = np.argmax(preds)
                emotion = emotion_labels[emotion_index]

                # Show prediction
                st.success(f"🧠 Detected Emotion for Player {idx+1}: **{emotion.capitalize()}**")
                st.bar_chart(preds)

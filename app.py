import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
model = load_model('/Users/seifeldenelmizayen/Desktop/Cellula_Computer Vision/Second week/teeth_classifier/teeth_vgg16_model.h5')

# Class labels
MOD_classes = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# Image Dimention
img_dim = 224


# App title
st.title("Mouth and Oral Disease Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255

    # Predict
    prediction = model.predict(img_array)
    predicted_class = MOD_classes[np.argmax(prediction)]

    st.success(f"Prediction: **{predicted_class}**")
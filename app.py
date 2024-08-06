import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('weed_identification_model.keras')

# Define class names
class_names = ['broadleaf', 'grass', 'soil', 'soybean']

# Streamlit app
st.title('Weed Identification and Control System')
st.write('By 20/47CS/01117 AINA TESTIMONY KEHINDE')
st.write("Upload an image to predict its class")

uploaded_file = st.file_uploader("Choose an image...", type="tif")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    
    st.write(f"Prediction: {predicted_class}")

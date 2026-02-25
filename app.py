import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("deep_leaf_model.h5")

st.title("Deep Leaf - Crop Disease Detection")
st.write("Upload a leaf image to detect crop disease.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.image(uploaded_file, caption="Uploaded Leaf Image")
    st.success(f"Predicted Class Index: {class_index}")
    st.info(f"Confidence: {confidence*100:.2f}%")
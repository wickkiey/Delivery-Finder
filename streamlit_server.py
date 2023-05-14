import streamlit as st
import numpy as np 
import os 
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from helper_funtions import get_cropped_images, predict_image_class



st.title("Delivery Finder")

st.write("Welcome to the app")

# @st.cache_data
def load_model():
    model_handle = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2'
    hub_model = hub.load(model_handle)
    
    clf_model = tf.keras.models.load_model('model/binary_22')
    return hub_model, clf_model


# get input image 

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    hub_model,clf_model = load_model()
    
    pil_image = Image.open(uploaded_file)

    # Convert the PIL image to a NumPy array
    image_array = np.array(pil_image)
    
    cropped_images, output_image = get_cropped_images(image_array,hub_model)
    
    classification_result = []
    
    for cropped_image in cropped_images:
        classification_result.append(predict_image_class(cropped_image,clf_model,thresh=.21))
    
    st.header("Detected Humans")
    st.image(output_image, caption='Uploaded Image.', use_column_width=True)
    
    if len(classification_result) > 0:
        st.header("Human Detected !")
    else:
        st.header("No Human Detected !")
    
    st.header("Brand Detected")
    st.write(classification_result)

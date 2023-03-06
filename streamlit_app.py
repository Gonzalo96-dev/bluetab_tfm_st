import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
# Installation keras y keras-preprocessing
import preprocessing
import pickle
import time

col1, col2, col3 = st.columns([1, 2, 1])

if 'image' not in st.session_state:
    st.session_state['image'] = 'not done'
def preprocess_image(im_path):
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    if im is None:
        print("La imagen no se ha leido correctamente: ", im_path)
        return None

    # Escalar los pixeles a un rango de 0 a 1
    # Redimensionar el tamaño de la imagen
    im = cv2.resize(im, (225, 225))
    return im


# BORRAR ESTA LINEA EN EL SIGUIENTE COMMIT
def predict(im):
    model = pickle.load("")
    test_image = preprocess_image(image)
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ['drink' 'food' 'inside' 'menu' 'outside']
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
        'drink': 0,
        'food': 0,
        'inside': 0,
        'menu': 0,
        'outside': 0,
    }
    result = f"{class_names[np.argmax(scores)]} with a {(100 * np.max(scores)).round(2)} percent confidence."
    return result


col1.header("What is the category? Yelp edition")
with col1.expander("Click for more information"):
    st.write('This model reads an image you upload of a restaurant and classifies it into five categories:'
             '- Food'
             '- Drinks'
             '- Exterior'
             '- Interior'
             '- Menu')

def change_image_state():
    st.session_state['image'] = 'done'

file_uploaded = col2.file_uploader("Choose file", type=['png', 'jpg', 'jpeg'], on_change= change_image_state)
camera_photo = col2.camera_input("Or take a photo yourself")


#Creamos una barra de progreso
progress_bar = col2.progress(0)
for perc_completed in range(100):
    time.sleep(0.05)
    progress_bar.progress(perc_completed + 1)

col2.success("Photo updated succesfully")



if file_uploaded is not None:
    fig, ax = plt.subplots()
    image = Image.open(file_uploaded)
    plt.imshow(image)
    predictions = predict(image)
    st.write(predictions)
    fig = plt.figure()
    st.pyplot(fig)

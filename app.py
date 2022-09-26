import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Fire And Smoke Detection")


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model


with st.spinner('Loading Model into Memory......'):
    model = load_model()

classes = ['Fire', 'Smoke']


def scale(image):
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return tf.image.resize(image, [224, 224])


def decode_img_url(image):
    img = tf.image.decode_jpeg(image, channels=3)
    img = scale(img)
    return np.expand_dims(img, axis=0)



menu = ["Image", 'URL']
choice = st.sidebar.selectbox("Menu", menu)


def load_image(image_file):
    img = Image.open(image_file)
    return img


# path = st.text_input('Enter Image URL TO classify...',
#                      'https://www.climatechangepost.com/media/news/2020/01/Extreme_wildfire_event.jpg.820x520_q95_crop-smart.jpg')


if (choice == "URL"):
    st.subheader("URL")
    path = st.text_input('Enter Image URL TO classify...')
    print(path)
    bt = st.button('Predict')
    if bt:
        content = requests.get(path).content  # conversion of URL to byte array
        st.write("Predicted Class:")
        with st.spinner('classifying......'):
            result = model.predict(decode_img_url(content))
            print(result)
            result = np.round(result[0][0],2)
            # 0 is fire; 1 is smoke
            if result > 0.5:
                st.write('Smoke')
            else:
                st.write('Fire')
            image = Image.open(BytesIO(content))
            st.image(image, caption='Classifying Image', use_column_width=True)

if (choice == "Image"):
    st.subheader("Image")
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

    # storing the uploaded file into static folder
    if image_file is not None:
        with open('./static/photo.jpg', 'wb') as f:
            f.write(image_file.getbuffer())

    bt_image = st.button('Predict')

    if bt_image:
        path = r'.\static\photo.jpg'
        test_image = image.load_img(path, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)

        with st.spinner('classifying......'):
            result = model.predict(test_image)
            print(result)
            result = np.round(result[0][0], 2)
            # 0 is fire; 1 is smoke
            if result > 0.5:
                st.write('Smoke')
            else:
                st.write('Fire')
        st.image(Image.open(path))


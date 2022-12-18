import imp
import streamlit as st
from PIL import Image
from func import detect_age, detect_emotion, detect_face_part, detect_gender, detect_race

st.title('Face Attributes Detection')

image_input = st.file_uploader('Upload image:')

if image_input is not None:
    with open('temp.jpg','wb') as f:
        f.write(image_input.read())

emotion = detect_emotion('temp.jpg')
age = detect_age('temp.jpg')
gender = detect_gender('temp.jpg')
race = detect_race('temp.jpg')

if st.button('Detect'):
    st.write('''---''')
    st.subheader('Attributes:')
    c1, c2 = st.columns(2)
    image = Image.open('temp.jpg')
    c1.image(image)

    c2.write(f'Age:`{age}`')
    c2.write(f'Gender: `{gender}`')
    c2.write(f'Emotions: `{emotion}`')
    c2.write(f'Race: `{race}`')

  



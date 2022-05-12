# Imports
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import streamlit as st
import matplotlib.pyplot as plt

label_dict = {0: 'Air Conditioner',
              1: 'Car Horn',
              2: 'Children Playing',
              3: 'Dog Bark',
              4: 'Drilling',
              5: 'Engine Idling',
              6: 'Gun Shot',
              7: 'Jackhammer',
              8: 'Siren',
              9: 'Street Music'
              }

model = tf.keras.models.load_model('audio_classification.hdf5')

st.title('Urban Sound Classification Model')

option = st.selectbox('Sounds:',
                      ('Sound 1', 'Sound 2', 'Sound 3', 'Sound 4', 'Sound 5', 'Sound 6', 'Sound 7', 'Sound 8', 'Sound 9', 'Sound 10'))
st.write('You selected:', option)

if option == "Sound 1":
    option = '30204-0-0-0.wav'
elif option == "Sound 2":
    option = '72259-1-9-11.wav'
elif option == "Sound 3":
    option = '72015-2-0-4.wav'
elif option == "Sound 4":
    option = '7965-3-16-0.wav'
elif option == "Sound 5":
    option = '76086-4-0-27.wav'
elif option == "Sound 6":
    option = '6988-5-0-5.wav'
elif option == "Sound 7":
    option = '77246-6-0-0.wav'
elif option == "Sound 8":
    option = '33340-7-11-0.wav'
elif option == "Sound 9":
    option = '22601-8-0-17.wav'
elif option == "Sound 10":
    option = '35548-9-2-7.wav'

audio, sample_rate = librosa.load(option, res_type='kaiser_fast')
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
melspectrogram =librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=40,fmax=8000)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)
predicted_label = model.predict(mfccs_scaled_features)
prediction_class = np.argmax(predicted_label, axis=1)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.power_to_db(melspectrogram,ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
    ax.set(title='Mel Spectogram')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.power_to_db(mfccs_features,ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
    ax.set(title='MFCC')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.tight_layout()
    st.pyplot(fig)

audio_file = open(option, 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/ogg')

col1, col2 = st.columns(2)

with col1:
    st.header("Model Prediction: ")
    st.write(label_dict[prediction_class[0]])
with col2:
    st.header("Actual Sound: ")
    st.write(label_dict[int(option.split("-")[-3])])
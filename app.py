import tensorflow as tf
from tensorflow.keras import models
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, request
from flask import Flask
from gevent.pywsgi import WSGIServer
import waitress

def get_spectrogram_and_label_id(audio_file, label):
    spectrogram = audio_file
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = file_path
    return waveform, label

def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds

def transformator(file_gambar_spectogram):
    IMG_PATH = file_gambar_spectogram
    image = cv2.imread(IMG_PATH)
    image = cv2.resize(image, (512, 512))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def identifikasi_gambar(gambar):
    y_pred = np.argmax(model.predict(transformator(gambar)), axis=1)
    if str(y_pred[0]) == "0":
       label_nama = "Depresi"
    elif str(y_pred[0]) == "1":
       label_nama = "Normal"
    return label_nama

def transform_audio_to_spectogram_to_testing(audio_file):
    y, sr = librosa.load(audio_file)
    librosa.feature.melspectrogram(y=y, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                               fmax=8000)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB)
            # ax.set(title='Log Mel-frequency spectrogram')
    folder_dituju = "./titip"
    savefigure = os.path.join(folder_dituju, 'jalur_lewat.jpg')
    plt.savefig(savefigure)
    status = identifikasi_gambar(savefigure)
    return status

AUTOTUNE = tf.data.AUTOTUNE

#audio_file = "E:\Capstone_Project\Depressionvsnormal\Depression\Depression4.wav"
#hasil = transform_audio_to_spectogram_to_testing(audio_file)

#print(hasil)

model_ml = "./predictive_model_v_8.h5"
model = models.load_model(model_ml)


print("sukses")
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def infer_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Masukan hanya jenis file saja"
        file = request.files.get('file')
        try:
            img = transform_audio_to_spectogram_to_testing(file)
            img_respon =  { "status_user":img,"status_running":"Sukses"}
        except:
            img_respon =  { "status_user":"-","status_running":"Gagal"}
        return img_respon
    
@app.route('/', methods=['GET'])
def index():
    return "Selamat datang dimachine learning kami"

if __name__ == '__main__':
     app.debug = False
     port = int(os.environ.get('PORT', 33507))
     waitress.serve(app, port=port)
    # Debug/Development
    # app.run(debug=True, host="0.0.0.0", port="5000")
    # Production
    #http_server = WSGIServer(('127.0.0.1', 5000), app)
    #http_server.serve_forever()


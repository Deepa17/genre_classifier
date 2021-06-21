import os
import numpy as np
#import matplotlib.pyplot as plt
import librosa
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import model_from_json
import pickle

from flask import flash, Flask, redirect, url_for, request, render_template,session
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key='12345678'
print('Visit http://127.0.0.1:5000')

UPLOAD_FOLDER = os.getcwd() + '/uploads'

#function to extract the required features
def extract_features(song):
    data,sample_rate = librosa.load(song)
    result = np.array([])
    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally
    
    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    return result

 #open the saved model      
json_file = open('model-4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

#load the weights
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model-4.h5")

#loading the scaler and encoder
scaler = pickle.load(open(f'scaler-4.pkl','rb'))
encoder = pickle.load(open(f'encoder-4.pkl','rb'))

#home page   
@app.route('/')
def upload_form():
    path = os.getcwd() + '/uploads'
    return render_template('index.html')#, songs=songs, models=models)

#route for posting the wav file
@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        else:# and file.split('.')[-1]=='wav':
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            flash('File successfully uploaded')
            return redirect(url_for('.predict', path=filename))

#prdicting the result
@app.route('/result')
def predict():
    test_file = request.args['path']
    test=[]
    test.append(extract_features(test_file))
    test = pd.DataFrame(test)

    test = scaler.transform(test)
    test = np.expand_dims(test, axis=2)
    label= loaded_model.predict(test)
    pred = encoder.inverse_transform(label)
    return render_template("result.html", result=pred[0][0], path=test_file)

    
if __name__ == '__main__':
    app.run(port=5000, debug=True)#,threaded=False)
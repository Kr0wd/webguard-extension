import pandas as pd
import numpy as np
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

vectorizer = joblib.load('local_vectorizer.pkl')
svm = joblib.load('local_svm_model.pkl')
tokenizer = joblib.load('local_tokenizer.pkl')
cnn_model = load_model('local_hybrid_model.keras')

def sp(u): 
    return re.sub(r'^www\.', '', re.sub(r'^https?://', '', str(u))).rstrip('/')

urls = ['http://montrealgazette.com/columnists/index.html', 'http://london-city-hotel.co.uk/html/londo/twins172.php']
for u in urls:
    c = sp(u)
    v = vectorizer.transform([c])
    s = tokenizer.texts_to_sequences([c])
    sq = pad_sequences(s, maxlen=500)
    print(u)
    print('SVM:', svm.predict_proba(v)[0])
    print('CNN:', cnn_model.predict(sq, verbose=0)[0])

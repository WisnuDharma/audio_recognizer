import streamlit as st
import os
import numpy as np
import librosa
import tensorflow as tf
import joblib
from utils import extract_features_full
import pandas as pd
from scipy.stats import mode

try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'audio_classifier.h5')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'label_encoder.pkl')
SAMPLE_RATE = 22050
N_MFCC = 40
AUDIO_DURATION = 3  # detik
WINDOW_SIZE = 0.5
WINDOW_STEP = 0.125

# Load deskripsi kelas dari CSV
DESC_CSV_PATH = os.path.join(os.path.dirname(__file__), 'description.csv')
FIXED_CSV_PATH = os.path.join(os.path.dirname(__file__), 'description_fixed.csv')
try:
    desc_df = pd.read_csv(DESC_CSV_PATH)
except Exception:
    try:
        desc_df = pd.read_csv(FIXED_CSV_PATH)
    except Exception:
        st.error('File deskripsi lagu tidak bisa dibaca. Pastikan file CSV sudah benar.')
        st.stop()
CLASS_DESCRIPTIONS = {
    row['class']: {
        'name_song': row['name_song'],
        'theme': row['theme'],
        'lyrics': row['lyrics'],
        'translation': row['translation']
    }
    for _, row in desc_df.iterrows()
}
LABEL_TO_NAME = {row['class']: row['name_song'] for _, row in desc_df.iterrows()}

def predict_audio(temp_path):
    y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    window_len = int(WINDOW_SIZE * SAMPLE_RATE)
    step_len = int(WINDOW_STEP * SAMPLE_RATE)
    features_list = []
    for start in range(0, len(y) - window_len + 1, step_len):
        y_win = y[start:start+window_len]
        features = extract_features_full(y_win, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        features_list.append(features)
    if not features_list:
        st.error('Audio terlalu pendek untuk diproses.')
        return
    X_pred = np.stack(features_list)
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        le = joblib.load(LABEL_ENCODER_PATH)
        preds = model.predict(X_pred)
        pred_indices = np.argmax(preds, axis=1)
        majority_idx = mode(pred_indices, keepdims=False).mode
        pred_class = le.inverse_transform([majority_idx])[0]
        avg_probs = np.mean(preds, axis=0)
        prob_dict = {le.inverse_transform([i])[0]: float(avg_probs[i]) * 100 for i in range(len(avg_probs))}
        prob_df = pd.DataFrame({
            'Jenis Lagu': [LABEL_TO_NAME.get(k, k) for k in prob_dict.keys()],
            'Probabilitas (%)': [f'{v:.2f}' for v in prob_dict.values()]
        })
        name_song = LABEL_TO_NAME.get(pred_class, pred_class)
        st.success(f'Jenis Lagu: {name_song}')
        desc = CLASS_DESCRIPTIONS.get(pred_class)
        if desc:
            st.markdown('---')
            st.write('Tema:', desc['theme'])
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Lirik')
                st.write(desc['lyrics'])
            with col2:
                st.subheader('Terjemahan')
                st.write(desc['translation'])
            st.markdown('---')
        else:
            st.info('Tidak ada deskripsi untuk kelas ini.')
        st.write('Probabilitas per kelas:')
        st.table(prob_df)
    except Exception as e:
        st.error(f'Gagal prediksi: {e}')

st.title('Baliness Folksong Audio Recognition')
st.header('Input Audio')
input_mode = st.radio('Pilih sumber audio:', ['Upload File', 'Rekam Mic (Browser)'])
audio_bytes = None

if input_mode == 'Upload File':
    predict_file = st.file_uploader('Upload audio untuk prediksi', type=['wav', 'mp3'], key='predict')
    if predict_file:
        audio_bytes = predict_file.read()
elif input_mode == 'Rekam Mic (Browser)':
    if AUDIO_RECORDER_AVAILABLE:
        st.info('Klik tombol di bawah untuk merekam audio dari browser, lalu prediksi otomatis.')
        audio_bytes = audio_recorder()
        if audio_bytes is not None and len(audio_bytes) == 0:
            audio_bytes = None
    else:
        st.warning('audio_recorder_streamlit belum terinstall. Install dengan: pip install audio_recorder_streamlit')

if audio_bytes:
    temp_path = 'temp_predict.wav'
    with open(temp_path, 'wb') as f:
        f.write(audio_bytes)
    predict_audio(temp_path)
    os.remove(temp_path) 
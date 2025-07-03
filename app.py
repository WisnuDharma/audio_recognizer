import streamlit as st
import os
import numpy as np
import librosa
import tensorflow as tf
import joblib
from utils import extract_features_full
import sounddevice as sd
import wavio
import time
from scipy.stats import mode
import pandas as pd

try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'audio_classifier.h5')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'label_encoder.pkl')
SAMPLE_RATE = 22050
N_MFCC = 40
AUDIO_DURATION = 3  # default durasi prediksi jika upload file
WINDOW_SIZE = 0.5  # detik (lebih kecil untuk deteksi lebih presisi)
WINDOW_STEP = 0.125  # detik (overlap 75%)

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
# Mapping label ke nama lagu
LABEL_TO_NAME = {row['class']: row['name_song'] for _, row in desc_df.iterrows()}

st.title('Baliness Folksong Audio Recognation')

st.header('Input Audio')
input_mode = st.radio('Choose Audio Source:', ['Upload File', 'Record'])
audio_bytes = None
predict_file = None

if input_mode == 'Upload File':
    predict_file = st.file_uploader('Upload audio for prediction', type=['wav', 'mp3'], key='predict')
    if predict_file:
        audio_bytes = predict_file.read()

elif input_mode == 'Record':

    # Pseudo real-time mic hanya muncul di sub mic
    st.header('Real-Time Mic Prediction (Setiap 3 Detik)')
    if 'realtime_mic' not in st.session_state:
        st.session_state['realtime_mic'] = False
    if 'last_pred' not in st.session_state:
        st.session_state['last_pred'] = None
    if 'last_desc' not in st.session_state:
        st.session_state['last_desc'] = None
    if 'last_prob' not in st.session_state:
        st.session_state['last_prob'] = None
    if 'last_prob_df' not in st.session_state:
        st.session_state['last_prob_df'] = None

    colA, colB = st.columns(2)
    with colA:
        if not st.session_state['realtime_mic']:
            if st.button('▶️ Mulai Mic Real-Time'):
                st.session_state['realtime_mic'] = True
                st.rerun()
        else:
            if st.button('⏹️ Stop'):
                st.session_state['realtime_mic'] = False
                st.rerun()

    if st.session_state['realtime_mic']:
        st.info('Mic aktif, prediksi akan update setiap 3 detik...')
        duration = 3
        audio = np.empty((int(SAMPLE_RATE * duration), 1), dtype=np.float32)
        def callback(indata, frames, time_info, status):
            callback.idx += frames
            if callback.idx < audio.shape[0]:
                audio[callback.idx-frames:callback.idx, 0] = indata[:, 0]
        callback.idx = 0
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
            time.sleep(duration)
        wavio.write('temp_predict.wav', audio, SAMPLE_RATE, sampwidth=2)
        y, sr = librosa.load('temp_predict.wav', sr=SAMPLE_RATE)
        # Normalisasi audio
        y = (y - np.mean(y)) / (np.std(y) + 1e-8)
        window_len = int(WINDOW_SIZE * SAMPLE_RATE)
        step_len = int(WINDOW_STEP * SAMPLE_RATE)
        features_list = []
        for start in range(0, len(y) - window_len + 1, step_len):
            y_win = y[start:start+window_len]
            features = extract_features_full(y_win, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
            features_list.append(features)
        if features_list:
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
                st.session_state['last_pred'] = pred_class
                st.session_state['last_desc'] = CLASS_DESCRIPTIONS.get(pred_class)
                st.session_state['last_prob'] = prob_dict
                st.session_state['last_prob_df'] = prob_df
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
                st.session_state['last_pred'] = None
                st.session_state['last_desc'] = None
                st.session_state['last_prob'] = None
                st.session_state['last_prob_df'] = None
                st.error(f'Gagal prediksi: {e}')
        os.remove('temp_predict.wav')
        # Refresh otomatis setiap 3 detik
        st.rerun()

    if st.session_state['last_pred']:
        st.success(f'Prediksi kelas: {st.session_state["last_pred"]}')
        desc = st.session_state['last_desc']
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
        st.write('Probabilitas per kelas:')
        if 'last_prob_df' in st.session_state and st.session_state['last_prob_df'] is not None:
            st.table(st.session_state['last_prob_df'])
        else:
            st.info('Belum ada hasil probabilitas.')

if audio_bytes:
    temp_path = 'temp_predict.wav'
    with open(temp_path, 'wb') as f:
        f.write(audio_bytes)
    # Load audio
    y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
    # Jika input dari mic, lakukan prediksi per window kecil
    if input_mode == 'Rekam dari Mic':
        window_len = int(WINDOW_SIZE * SAMPLE_RATE)
        step_len = int(WINDOW_STEP * SAMPLE_RATE)
        features_list = []
        # Normalisasi audio (zero mean, unit variance)
        y = (y - np.mean(y)) / (np.std(y) + 1e-8)
        for start in range(0, len(y) - window_len + 1, step_len):
            y_win = y[start:start+window_len]
            features = extract_features_full(y_win, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
            features_list.append(features)
        if not features_list:
            st.error('Rekaman terlalu pendek untuk diproses.')
        else:
            X_pred = np.stack(features_list)
            try:
                model = tf.keras.models.load_model(MODEL_PATH)
                le = joblib.load(LABEL_ENCODER_PATH)
                preds = model.predict(X_pred)
                pred_indices = np.argmax(preds, axis=1)
                # Voting mayoritas
                majority_idx = mode(pred_indices, keepdims=False).mode
                pred_class = le.inverse_transform([majority_idx])[0]
                # Probabilitas rata-rata per kelas
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
    else:
        # Untuk upload file, pipeline seperti biasa
        target_len = AUDIO_DURATION * SAMPLE_RATE
        if len(y) > target_len:
            y = y[:target_len]
        elif len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        try:
            features = extract_features_full(temp_path, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
            X_pred = np.expand_dims(features, axis=0)
            model = tf.keras.models.load_model(MODEL_PATH)
            le = joblib.load(LABEL_ENCODER_PATH)
            pred = model.predict(X_pred)
            pred_indices = np.argmax(pred, axis=1)
            # Voting mayoritas
            majority_idx = mode(pred_indices, keepdims=False).mode
            pred_class = le.inverse_transform([majority_idx])[0]
            # Probabilitas rata-rata per kelas
            avg_probs = np.mean(pred, axis=0)
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
    os.remove(temp_path) 
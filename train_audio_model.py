import os
import sys
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Import extract_features_full dengan fallback
if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import extract_features_full
else:
    from .utils import extract_features_full

AUDIO_DIR = 'audio_data'
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'audio_classifier.h5')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
SAMPLE_RATE = 22050
N_MFCC = 40

# 1. Ekstraksi fitur gabungan dari semua file audio
def extract_features(file_path):
    return extract_features_full(file_path, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

# 2. Load data dan label
def load_data(audio_dir):
    features = []
    labels = []
    for label in os.listdir(audio_dir):
        label_dir = os.path.join(audio_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if not fname.lower().endswith('.wav'):
                continue
            file_path = os.path.join(label_dir, fname)
            try:
                feat = extract_features(file_path)
                features.append(feat)
                labels.append(label)
            except Exception as e:
                print(f'Gagal ekstrak {file_path}: {e}')
    return np.array(features), np.array(labels)

# 3. Training model
def train_model(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    input_shape = X.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=40, batch_size=16, validation_data=(X_test, y_test))
    return model, le

# 4. Simpan model dan label encoder
def save_model(model, le):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f'Model disimpan di {MODEL_PATH}')
    print(f'Label encoder disimpan di {LABEL_ENCODER_PATH}')

if __name__ == '__main__':
    print('Ekstraksi fitur audio...')
    X, y = load_data(AUDIO_DIR)
    print(f'Jumlah data: {len(X)}')
    print('Training model...')
    model, le = train_model(X, y)
    save_model(model, le)
    print('Selesai training!') 
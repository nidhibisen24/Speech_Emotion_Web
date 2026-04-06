import os
import json
import pickle
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

WEIGHTS_PATH = os.path.join(MODEL_DIR, "emotion_model_weights.weights.h5")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

MAX_PAD_LEN = config.get("max_pad_len", 174)
N_MFCC = config.get("n_mfcc", 40)

def build_model(input_shape=(174, 180), num_classes=6):
    model = Sequential([
        Input(shape=input_shape),

        Conv1D(64, kernel_size=5, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        Conv1D(128, kernel_size=5, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        Conv1D(256, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),

        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model(input_shape=(MAX_PAD_LEN, 180), num_classes=len(label_encoder.classes_))
model.load_weights(WEIGHTS_PATH)

def extract_features(file_path, max_pad_len=MAX_PAD_LEN, n_mfcc=N_MFCC):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)

        features = np.vstack([mfcc, chroma, mel]).T

        T = features.shape[0]
        if T < max_pad_len:
            pad_width = max_pad_len - T
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
        else:
            features = features[:max_pad_len, :]

        features = scaler.transform(features)
        features = np.expand_dims(features, axis=0).astype(np.float32)

        return features

    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {str(e)}")

def predict_emotion(file_path):
    features = extract_features(file_path)
    predictions = model.predict(features, verbose=0)[0]

    predicted_index = np.argmax(predictions)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(np.max(predictions) * 100)

    class_probabilities = {
        label_encoder.inverse_transform([i])[0]: float(predictions[i] * 100)
        for i in range(len(predictions))
    }

    return predicted_label, confidence, class_probabilities
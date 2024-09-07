import os
import onnxruntime as ort
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def evaluate_file_onnx(file_path, ort_session, label_encoder, button_encoder, scaler, segment_size=1000, overlap_size=500):
    df = pd.read_csv(file_path, header=None, dtype={0: float, 1: float, 2: float, 3: float, 4: float, 5: str})
    df[5] = button_encoder.transform(df[5])
    df[[0, 3, 4]] = scaler.transform(df[[0, 3, 4]])
    segments = []

    #24000 - 1000 + 1 = 23001
    #23001 + 500 = 23501
    for start in range(0, len(df) - segment_size + 1, segment_size - overlap_size):
        end = start + segment_size
        segment = df.iloc[start:end].values
        #PADDING LOGIC
        if len(segment) < segment_size:
            padding = np.zeros((segment_size - len(segment), segment.shape[1]))
            segment = np.vstack((segment, padding))

        segments.append(segment)
#not reeally useefull
    if not segments:
        print(f"No valid segments found in {file_path}.")
        return None
    predictions = []
    for segment in segments:
        segment = segment.astype(np.float32)
        segment = np.expand_dims(segment, axis=0)
        ort_inputs = {ort_session.get_inputs()[0].name: segment}
        ort_outs = ort_session.run(None, ort_inputs)
        predictions.append(ort_outs[0])
    avg_prediction = np.mean(predictions, axis=0)
    predicted_class = np.argmax(avg_prediction)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label
#RELEVANT STUFF BELOW
def classifyfilesinfolder(folder_path, onnx_model_path='osu_anti_cheat_model_best.onnx', label_encoder_path='label_encoder.pkl', button_encoder_path='button_encoder.pkl', scaler_mean_path='scaler_mean.pkl', scaler_std_path='scaler_std.pkl'):
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    with open(button_encoder_path, 'rb') as f:
        button_encoder = pickle.load(f)
    with open(scaler_mean_path, 'rb') as f_mean, open(scaler_std_path, 'rb') as f_std:
        scaler_mean = pickle.load(f_mean)
        scaler_std = pickle.load(f_std)
        scaler = StandardScaler()
        scaler.mean_ = scaler_mean
        scaler.scale_ = scaler_std
    ort_session = ort.InferenceSession(onnx_model_path)
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                predicted_label = evaluate_file_onnx(file_path, ort_session, label_encoder, button_encoder, scaler)
                print(f"File: {file_name} -> Predicted Label: {predicted_label}")
    print(button_encoder.classes_)
# Example usage
folder_path = '/mnt/4b55a907-bb3f-43f9-ad49-8c8f30f6a000/relaxtest'
classifyfilesinfolder(folder_path)

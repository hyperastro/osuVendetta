import os
import time
import onnxruntime as ort
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import torch


def load_model_and_helpers(onnx_model_path, encoder_paths):
    # Load ONNX runtime session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_model_path, providers=providers)

    # Load encoders and scaler
    with open(encoder_paths['label_encoder'], 'rb') as f:
        label_encoder = pickle.load(f)
    with open(encoder_paths['button_encoder'], 'rb') as f:
        button_encoder = pickle.load(f)
    with open(encoder_paths['scaler_mean'], 'rb') as f_mean, open(encoder_paths['scaler_std'], 'rb') as f_std:
        scaler = StandardScaler()
        scaler.mean_ = pickle.load(f_mean)
        scaler.scale_ = pickle.load(f_std)

    return session, label_encoder, button_encoder, scaler


# Preprocess the .txt file
def preprocess_file(file_path, button_encoder, scaler, segment_size=1000):
    try:
        df = pd.read_csv(file_path, header=None, dtype={0: float, 1: float, 2: float, 3: float, 4: float, 5: str})
        df[5] = button_encoder.transform(df[5])
        df[[0, 1, 2, 3, 4]] = scaler.transform(df[[0, 1, 2, 3, 4]])

        segments = []
        for start in range(0, len(df) - segment_size + 1, segment_size // 2):
            end = start + segment_size
            segment = df.iloc[start:end].values
            if len(segment) < segment_size:
                padding = np.zeros((segment_size - len(segment), segment.shape[1]))
                segment = np.vstack((segment, padding))
            segments.append(torch.tensor(segment, dtype=torch.float32))

        if not segments:
            return None
        return pad_sequence(segments, batch_first=True, padding_value=0).numpy()

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


# Perform inference on a single file using ONNX model
def classify_file(session, file_path, button_encoder, scaler, segment_size=1000):
    segments = preprocess_file(file_path, button_encoder, scaler, segment_size)
    if segments is None:
        print("File preprocessing failed.")
        return None

    input_name = session.get_inputs()[0].name
    h0_name = session.get_inputs()[1].name
    c0_name = session.get_inputs()[2].name

    # Initialize h0 and c0
    batch_size = segments.shape[0]
    hidden_size = 128
    num_layers = 3
    h0 = np.zeros((num_layers * 2, batch_size, hidden_size), dtype=np.float32)
    c0 = np.zeros((num_layers * 2, batch_size, hidden_size), dtype=np.float32)

    # Perform inference
    st = time.time()
    outputs = session.run(None, {input_name: segments, h0_name: h0, c0_name: c0})
    et = time.time()
    tt = et - st
    print(f"Prediction speed: {tt}")
    predictions = torch.sigmoid(torch.tensor(outputs[0])).squeeze().numpy()
    binary_preds = (predictions > 0.5).astype(int)
    return binary_preds


# Main inference function to process all .txt files in a directory
def main(directory_path):
    onnx_model_path = 'osuanticheatmodelbest.onnx'
    encoder_paths = {
        'label_encoder': 'label_encoder.pkl',
        'button_encoder': 'button_encoder.pkl',
        'scaler_mean': 'scaler_mean.pkl',
        'scaler_std': 'scaler_std.pkl'
    }

    # Load ONNX model and encoders
    session, label_encoder, button_encoder, scaler = load_model_and_helpers(onnx_model_path, encoder_paths)
    # Process each .txt file in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory_path, file_name)
            prediction = classify_file(session, file_path, button_encoder, scaler)
            if prediction is not None:
                decoded_labels = label_encoder.inverse_transform(prediction)
                print(f"Predicted classes for segments in {file_name}: {decoded_labels}")
            else:
                print(f"Prediction failed for file {file_name}")


# Usage: specify the directory containing .txt files
if __name__ == "__main__":
    directory_path = '/mnt/4b55a907-bb3f-43f9-ad49-8c8f30f6a000/C#replayparser/out'  # Replace with the actual directory path
    main(directory_path)


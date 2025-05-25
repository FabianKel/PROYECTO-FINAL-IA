import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd

def extract_important_features(file_path, offset=None, duration=None):
    # Cargar el archivo de audio
    y, sr = librosa.load(file_path, sr=None, offset=offset, duration=duration)
    S = librosa.stft(y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spec_centroid)
    spectral_centroid_var = np.var(spec_centroid)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spec_bw)
    spectral_bandwidth_var = np.var(spec_bw)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)
    zcr = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = np.mean(zcr)
    zero_crossing_rate_var = np.var(zcr)
    harmony = librosa.effects.harmonic(y)
    harmony_mean = np.mean(harmony)
    harmony_var = np.var(harmony)
    perceptr = librosa.feature.melspectrogram(y=y, sr=sr)
    perceptr_mean = np.mean(librosa.power_to_db(perceptr, ref=np.max))
    perceptr_var = np.var(librosa.power_to_db(perceptr, ref=np.max))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = [np.mean(m) for m in mfccs]
    mfcc_vars = [np.var(m) for m in mfccs]

    features = {
        #'song_title': os.path.basename(file_path),
        'chroma_stft_mean': chroma_stft_mean,
        'chroma_stft_var': chroma_stft_var,
        'rms_mean': rms_mean,
        'rms_var': rms_var,
        'spectral_centroid_mean': spectral_centroid_mean,
        'spectral_centroid_var': spectral_centroid_var,
        'spectral_bandwidth_mean': spectral_bandwidth_mean,
        'spectral_bandwidth_var': spectral_bandwidth_var,
        'rolloff_mean': rolloff_mean,
        'rolloff_var': rolloff_var,
        'zero_crossing_rate_mean': zero_crossing_rate_mean,
        'zero_crossing_rate_var': zero_crossing_rate_var,
        'harmony_mean': harmony_mean,
        'harmony_var': harmony_var,
        'perceptr_mean': perceptr_mean,
        'perceptr_var': perceptr_var,
        'tempo': tempo,
    }
    # Agregar MFCCs
    for i in range(20):
        features[f'mfcc{i+1}_mean'] = mfcc_means[i]
        features[f'mfcc{i+1}_var'] = mfcc_vars[i]

    return features

def extract_features(file_path, segment_duration=3.0):
    """Extrae un DataFrame con una fila por cada segmento del audio."""
    y, sr = librosa.load(file_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)

    features_list = []
    for start in np.arange(0, total_duration, segment_duration):
        segment_features = extract_important_features(file_path, offset=start, duration=segment_duration)
        features_list.append(segment_features)

    df = pd.DataFrame(features_list)
    return df

def generate_spectrogram(audio_path, output_dir):
    """Generate and save a spectrogram image for the entire file."""
    y, sr = librosa.load(audio_path)
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    
    output_path = output_dir / f"spectrogram_{Path(audio_path).stem}.png"
    plt.savefig(output_path)
    plt.close()
    return output_path
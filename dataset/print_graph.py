import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torch
from scipy.io import wavfile

sample_rate, waveform = wavfile.read('../dataset/harmonica_test/kou.wav')
# waveform, sample_rate = torchaudio.load("../dataset/harmonica_test/kou.wav")
# waveform = waveform.numpy()[0]
# waveform = waveform[np.newaxis, ...]  
# waveform = torch.from_numpy(waveform)	

# mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform) 
# mel_specgram = mel_specgram.detach().numpy()
# mel_specgram = mel_specgram[0, ...]

plt.specgram(waveform, Fs=sample_rate)
plt.title("Double mel_specgram (Heatmap)")
plt.xlabel('Sec (s)')
plt.ylabel('Frequency')
plt.savefig('../graph/kou_2d.png')
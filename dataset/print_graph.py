import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torch

waveform, sample_rate = torchaudio.load("../dataset/harmonica_test/double_graph.wav")
waveform = waveform.numpy()[0, :52876]
waveform = waveform[np.newaxis, ...]  
waveform = torch.from_numpy(waveform)	

mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform) 
mel_specgram = mel_specgram.detach().numpy()
mel_specgram = mel_specgram[0, ...]

plt.imshow(mel_specgram)
plt.title("Double mel_specgram (Heatmap)")
plt.xlabel('Sec (s)')
plt.ylabel('Frequency')
plt.savefig('../graph/double_2d.png')
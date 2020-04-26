from scipy.io import wavfile
sample_rate, waveform = wavfile.read("record.wav")
waveform = waveform[..., 0]

print(waveform[0: 50000])
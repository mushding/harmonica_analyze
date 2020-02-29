import torch
import torch.nn as nn
import numpy as np
import torchaudio
from load_model import CNN, predict 

track_array = []
FRAME_SIZE = 5200
STEP_SIZE = 4000

# read sound track to predict
waveform, sample_rate = torchaudio.load("../../dataset/harmonica_track/flat.wav")	        
new_sample_rate = sample_rate / 4
waveform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[0, :].view(1, -1))  
        
waveform = waveform.numpy()[0, :]
length = np.shape(waveform)[0] 

for index in range(0, length, STEP_SIZE):
    waveform_part = waveform[index: index + FRAME_SIZE]													                
    waveform_part = waveform_part[np.newaxis, ...]                                              
    waveform_part = torch.from_numpy(waveform_part)										
    waveform_part = waveform_part.detach().numpy()
    if np.shape(waveform_part)[1] == FRAME_SIZE:
        track_array.append(waveform_part)
tensor_track = torch.Tensor(track_array)

# load model
model = torch.load('../../model/harmonica_model/harmonica_error_model.pth')
print("loading model...")
print('-'*50)

# put data in cnn
print("predicting...")
print('-'*50)
output = predict(tensor_track, model)
print(output)





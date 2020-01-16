#! /usr/bin/env python
######################################################################
# tuner.py - a minimal command-line guitar/ukulele tuner in Python.
# Requires numpy and pyaudio.
######################################################################
# Author:  Matt Zucker
# Date:    July 2016
# License: Creative Commons Attribution-ShareAlike 3.0
#          https://creativecommons.org/licenses/by-sa/3.0/us/
######################################################################

import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import torch
import torch.nn as nn

######################################################################
# Feel free to play with these numbers. Might want to change NOTE_MIN
# and NOTE_MAX especially for guitar/bass. Probably want to keep
# FRAME_SIZE and FRAMES_PER_FFT to be powers of two.

NOTE_MIN = 0       # C4
NOTE_MAX = 100       # A4
FSAMP = 11025       # Sampling frequency in Hz
FRAME_SIZE = 1024   # How many samples per frame?
FRAMES_PER_FFT = 8 # FFT takes average across how many frames?
INPUT_DEVICE = 1

######################################################################
# Derived quantities from constants above. Note that as
# SAMPLES_PER_FFT goes up, the frequency step size decreases (so
# resolution increases); however, it will incur more delay to process
# new sounds.

# SAMPLES_PER_FFT = FRAME_SIZE*FRAMES_PER_FFT
SAMPLES_PER_FFT = 16538
FREQ_STEP = float(FSAMP)/SAMPLES_PER_FFT

######################################################################

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (10, 1, 66150) (batchsize, in_chennel, time_stamps)
            nn.Conv1d(
                in_channels=1,         
                out_channels=16,      
                kernel_size=20,        
                stride=1,              
            ),                         
            nn.Tanh(),                 
            nn.MaxPool1d(kernel_size=5), 
        )
        self.conv2 = nn.Sequential(      
            nn.Conv1d(16, 32, 20, 1), 
            nn.Tanh(),     
            nn.MaxPool1d(kernel_size=5),              
        )
        self.conv3 = nn.Sequential(      
            nn.Conv1d(32, 64, 20, 1), 
            nn.Tanh(),     
            nn.MaxPool1d(kernel_size=5),              
        )
        self.out = nn.Sequential(
            nn.Linear(8128, 1000),
            nn.Tanh(),
            nn.Linear(1000, 5),   # fully connected layer, output 10 classes
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

######################################################################

# Allocate space to run an FFT. 
buf = np.zeros(SAMPLES_PER_FFT, dtype=np.float32)
num_frames = 0

# Initialize audio
stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,
                                channels=1,
                                rate=FSAMP,
                                input=True,
                                frames_per_buffer=FRAME_SIZE,
                                input_device_index=INPUT_DEVICE)

stream.start_stream()

# Create Hanning window function
window = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, SAMPLES_PER_FFT, False)))

# Print initial text
print ('sampling at', FSAMP, 'Hz with max resolution of', FREQ_STEP, 'Hz')

# As long as we are getting data:
while stream.is_active():

    # Shift the buffer down and new data in
    buf[:-FRAME_SIZE] = buf[FRAME_SIZE:]
    buf[-FRAME_SIZE:] = np.fromstring(stream.read(FRAME_SIZE), np.int16) / 32767
    buf = buf[np.newaxis, ...]

    tensor_test = torch.Tensor(buf)

    model = torch.load('./model/harmonica_model_5_resample_best.pth')
    print('-'*50)

    predict = model(tensor_test)
    pred_y = torch.max(predict, 1)[1].data.numpy()
    print(pred_y)

    
    

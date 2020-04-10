import numpy as np
import os
import torchaudio
import torch


SECOND = 0.5
SAMPLE_FREQUANCE = 44100
FRAME_SIZE = int(SAMPLE_FREQUANCE * SECOND)
STEP_SIZE = 20000

CHECK_SIZE = 4410   # 0.1 sec
CHECK_NUM = 4000

name_dir = "temp"
filename = os.listdir("./record/" + name_dir)
filenum = len(filename) 

# read sound track to predict
for file in range(filenum):
    waveform, sample_rate = torchaudio.load("./record/" + name_dir + "/" + filename[file]) # torch.Size([1, x])        
    length = np.shape(waveform)[1]
    num = 1
    for index in range(0, length, STEP_SIZE):
        waveform_part = waveform[:, index: index + FRAME_SIZE]													                
        if np.shape(waveform_part)[1] == FRAME_SIZE:
            torchaudio.save("./harmonica_dataset/wind/" + str(num) + "_" + filename[file], waveform_part, sample_rate)
            num = num + 1
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
CUT_LIMIT = 0
# CUT_LIMIT = 0.05

# name_dir_array = ['flat', 'double', 'normal'] 
name_dir_array = ['space'] 
for name_dir in name_dir_array:
    filename = os.listdir("./record/" + name_dir)
    filenum = len(filename) 
    # read sound track to predict
    for file in range(filenum):
        cut_waveform = []
        waveform, sample_rate = torchaudio.load("./record/" + name_dir + "/" + filename[file]) # torch.Size([1, x])
        print(filename[file], "before cutting:", waveform.size())
        waveform = list(waveform[0].numpy())            # tensor -> ndarray -> list
        for index in range(0, len(waveform), CHECK_SIZE):
            num = 0
            check_waveform = waveform[index: index + CHECK_SIZE]
            for check in check_waveform:
                if check < CUT_LIMIT and check > -CUT_LIMIT: 
                    num = num + 1
            if num < CHECK_NUM:                              # the number of smaller than 0.03
                cut_waveform.extend(check_waveform)

        cut_waveform = np.array(cut_waveform)           # list -> ndarray  
        cut_waveform = cut_waveform[np.newaxis, ...]    # add new axis
        cut_waveform = torch.from_numpy(cut_waveform)   # ndarray -> tensor
        print(filename[file], "after cutting:", cut_waveform.size())
        length = np.shape(cut_waveform)[1]
        num = 1
        for index in range(0, length, STEP_SIZE):
            waveform_part = cut_waveform[:, index: index + FRAME_SIZE]													                
            if np.shape(waveform_part)[1] == FRAME_SIZE:
                torchaudio.save("./harmonica_dataset/" + name_dir + "/" + str(num) + "_" + filename[file], waveform_part, sample_rate)
                num = num + 1
from scipy.io import wavfile
import numpy as np

class Cut:
    def cut_start(self):
        CHECK_SIZE = 4410
        CUT_LIMIT = 2000
        CHECK_NUM = 4100

        sample_rate, waveform = wavfile.read("./static/HarmonicaData/wav/record.wav")
        waveform = waveform[..., 0]
        print(np.shape(waveform))
        start_index = 0
        for index in range(0, len(waveform), CHECK_SIZE):
            num = 0
            check_waveform = waveform[index: index + CHECK_SIZE]
            for check in check_waveform:
                if check < CUT_LIMIT and check > -CUT_LIMIT: 
                    num = num + 1
            print(num)
            if num > CHECK_NUM:                              
                start_index = index + CHECK_SIZE
            else:
                break 
        print(start_index)       
        wavfile.write("./static/HarmonicaData/wav/record.wav", sample_rate, waveform[start_index: ])
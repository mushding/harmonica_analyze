import numpy as np
import torchaudio
import torch

FRAME_SIZE = 22050  # 0.25 sec
STEP_SIZE = 20000
CHECK_SIZE = 4410   # 0.1 sec
CUT_LIMIT = 0.05

name_dir_array = ['flat', 'normal', 'double'] 
for name_dir in name_dir_array:
    read_file = [name_dir + '_do_mid', name_dir + '_re_mid', name_dir + '_mi_mid', name_dir + '_fa_mid', name_dir + '_so_mid', name_dir + '_la_mid', name_dir + '_si_mid', name_dir + '_do_high']
    save_file = [name_dir + '_do_mid', name_dir + '_re_mid', name_dir + '_mi_mid', name_dir + '_fa_mid', name_dir + '_so_mid', name_dir + '_la_mid', name_dir + '_si_mid', name_dir + '_do_high']
    # read sound track to predict
    for read, save in zip(read_file, save_file):
        cut_waveform = []
        waveform, sample_rate = torchaudio.load("./harmonica_track/" + read + ".wav") # torch.Size([1, x])
        print(read, "before cutting:", waveform.size())
        waveform = list(waveform[0].numpy())            # tensor -> ndarray -> list
        for index in range(0, len(waveform), CHECK_SIZE):
            num = 0
            check_waveform = waveform[index: index + CHECK_SIZE]
            for check in check_waveform:
                if check < CUT_LIMIT and check > -CUT_LIMIT: 
                    num = num + 1
            if num < 4000:                              # the number of smaller than 0.03
                cut_waveform.extend(check_waveform)

        cut_waveform = np.array(cut_waveform)           # list -> ndarray  
        cut_waveform = cut_waveform[np.newaxis, ...]    # add new axis
        cut_waveform = torch.from_numpy(cut_waveform)   # ndarray -> tensor
        print(read, "after cutting:", cut_waveform.size())
        length = np.shape(cut_waveform)[1]
        num = 1
        for index in range(0, length, STEP_SIZE):
            waveform_part = cut_waveform[:, index: index + FRAME_SIZE]													                
            if np.shape(waveform_part)[1] == FRAME_SIZE:
                torchaudio.save("./harmonica_dataset/" + name_dir + "/" + save + "_" + str(num) + ".wav", waveform_part, sample_rate)
                num = num + 1
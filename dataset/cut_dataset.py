# import matplotlib
# matplotlib.use('Agg')
import torchaudio
import numpy as np
import torch
import scipy.io.wavfile
# import matplotlib.pyplot as plt

CUT_LIMIT = 0.03
cut_waveform = []

waveform, semple_rate = torchaudio.load("./double_re.wav")  # torch.Size([1, x])
waveform = list(waveform[0].numpy())        # tensor -> ndarray -> list

# plt.plot(waveform)
# plt.savefig('no_cut.png')

for node in waveform:
    if node > CUT_LIMIT or node < -CUT_LIMIT:
        cut_waveform.append(node)

# plt.plot(cut_waveform)
# plt.savefig('cut.png')

cut_waveform = np.array(cut_waveform)       # list -> ndarray  
cut_waveform = cut_waveform[np.newaxis, ...]    # add new axis
cut_waveform = torch.from_numpy(cut_waveform)   # ndarray -> tensor

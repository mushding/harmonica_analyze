import torch
import torch.nn as nn
import numpy as np
import torchaudio
from load_model import CNN ,predict 


class Open_track:
    def __init__(self):
        self.track_array = []
        self.RESEMPLE_RATE = 1
        self.SECOND = 0.5
        self.SAMPLE_FREQUANCE = 44100
        self.FRAME_SIZE = int((self.SAMPLE_FREQUANCE * self.SECOND) / self.RESEMPLE_RATE)
        self.STEP_SIZE = 4410
        self.SECONDforarray = 0.1

    def readsound(self, wav_name):    
        # read sound track to predict
        #args = parseArguments()
        waveform, sample_rate = torchaudio.load("./static/HarmonicaData/wav/" + str(wav_name))	        
        new_sample_rate = sample_rate / self.RESEMPLE_RATE   #turn to 1
        waveform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[0, :].view(1, -1))  

        waveform = waveform.numpy()[0, :]
        length = np.shape(waveform)[0] 

        for index in range(0, length, self.STEP_SIZE):
            waveform_part = waveform[index: index + int(self.FRAME_SIZE)]                            # [44100]             
            if len(waveform_part) < self.FRAME_SIZE:
                break
            waveform_part = waveform_part[np.newaxis, ...]                                      # [1, 44100]      
            waveform_part = torch.from_numpy(waveform_part)	                                    # torch [1, 44100]
            #print(np.shape(waveform_part))
            mel_specgram = torchaudio.transforms.MelSpectrogram(new_sample_rate)(waveform_part) # torch [1, 128, 221]
            mel_specgram = mel_specgram.detach().numpy()					                    # numpy [1, 128, 221]																	                
            
            if np.shape(waveform_part)[1] == int(self.FRAME_SIZE):
                self.track_array.append(mel_specgram)

        tensor_track = torch.Tensor(self.track_array)

        return tensor_track

    def loadmodel(self):
        # load model
        model = torch.load('./model/harmonica_error_2d_model_10.pth')
        print("loading model...")
        print('-'*50)

        return model

    def putincnn(self, tensor_track, model):
        # put data in cnn
        print("predicting...")
        print('-'*50)
        output = predict(tensor_track, model)
        print(output)
        return output

    

    def findwrongnote(self,output):
        outputarray = []
        Dout ={"start":0, "end":0,'drag': False, 'resize': False, "type":'0'}
        skip_until = -1
        for index_i in range(len(output)):
            
            if  skip_until >= index_i:
                continue       

            if output[index_i] == 1:
                Dout["start"] = round((self.SECONDforarray * index_i),1)
                for index_j in range(len(output[index_i:])):
                    if output[index_i + index_j] == output[index_i]:
                        index_j+=1
                    else:
                        Dout["end"] = round(self.SECONDforarray * (index_i + index_j),1)
                        Dout["type"] = '1'
                        skip_until = (index_i + index_j)
                        outputarray.append(dict(Dout))
                        break

            if output[index_i] == 2:
                Dout["start"] = round((self.SECONDforarray * index_i),1)
                for index_j in range(len(output[index_i:])):
                    if output[index_i + index_j] == output[index_i]:
                        index_j+=1
                    else:                        
                        Dout["end"] = round(self.SECONDforarray * (index_i + index_j),1)
                        Dout["type"] = '2'
                        skip_until = (index_i + index_j)                       
                        outputarray.append(dict(Dout))                        
                        break
              

        print(outputarray)
        return outputarray




                










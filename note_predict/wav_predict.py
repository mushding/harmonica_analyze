<<<<<<< HEAD
import torch
import torch.nn as nn
import numpy as np
import torchaudio

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

test_array = []
for i in range(1, 20):
    test_waveform, sample_rate = torchaudio.load('../dataset/harmonica_note_test_dataset/harmonica_note_test/test_' + str(i) + '.wav')      # test data
    new_sample_rate = sample_rate/4
    test_waveform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(test_waveform[0,:].view(1, -1))
    test_waveform = test_waveform.numpy()[0, :]
    for i in test_waveform:
        i = int(i)    
    test_waveform = test_waveform[np.newaxis, ...]                                          
    test_waveform = torch.from_numpy(test_waveform)

    test_waveform = test_waveform.detach().numpy()
    test_array.append(test_waveform)

tensor_test = torch.Tensor(test_array)
print(np.shape(tensor_test))

model = torch.load('../model/harmonica_note_test_model/harmonica_model_5_resample_best.pth')
print(model)
print('-'*50)

predict = model(tensor_test)
print(predict)
pred_y = torch.max(predict, 1)[1].data.numpy()
print(pred_y + 1, 'prediction number')
=======
import torch
import torch.nn as nn
import numpy as np
import torchaudio

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

test_array = []
for i in range(1, 20):
    test_waveform, sample_rate = torchaudio.load('../dataset/harmonica_note_test_dataset/harmonica_note_test/test_' + str(i) + '.wav')      # test data
    new_sample_rate = sample_rate/4
    test_waveform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(test_waveform[0,:].view(1, -1))
    test_waveform = test_waveform.numpy()[0, :]
    for i in test_waveform:
        i = int(i)    
    test_waveform = test_waveform[np.newaxis, ...]                                          
    test_waveform = torch.from_numpy(test_waveform)

    test_waveform = test_waveform.detach().numpy()
    test_array.append(test_waveform)

tensor_test = torch.Tensor(test_array)
print(np.shape(tensor_test))

model = torch.load('../model/harmonica_note_test_model/harmonica_model_5_resample_best.pth')
print(model)
print('-'*50)

predict = model(tensor_test)
print(predict)
pred_y = torch.max(predict, 1)[1].data.numpy()
print(pred_y + 1, 'prediction number')
>>>>>>> 6789b8d695be03539abd3064bdb9efd70bf530da
print('[1 2 5 4 1 4 5 3 2 1 4 2 5 4 5 1 1 2 3] real number')
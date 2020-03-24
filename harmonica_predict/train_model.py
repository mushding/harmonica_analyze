import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import torchaudio
import matplotlib.pyplot as plt

# Hyper Parameters
BATCH_SIZE = 150  # num of training examples per minibatch
EPOCH = 30
LR = 0.001

# dataset Hyper Parameters
RESAMPLE_RATE = 1
SECOND = 0.5
SAMPLE_FREQUANCE = 44100
FRAME_SIZE = int((SAMPLE_FREQUANCE * SECOND) / RESAMPLE_RATE)
STEP_SIZE = 5000

index_array = []
note_array = []
test_array = []
dataset = ["space", "flat", "double", "normal"]

os.chdir("../dataset/harmonica_dataset")
for index, condition in enumerate(dataset):               
    filename = os.listdir(condition)                                                                        # training data
    filenum = len(filename)
    print("read wav file : " + condition)
    for file in range(filenum):
        print("processing (data) " + condition + " " + str(file))
        waveform, sample_rate = torchaudio.load(condition + '/' + filename[file])	                            # read waveform shape [2, 21748]
        new_sample_rate = sample_rate / RESAMPLE_RATE
        waveform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[0,:].view(1,-1))   # shape [1, 5437]
        waveform = waveform.numpy()[0, :int(FRAME_SIZE)]													# shape [5437]

        waveform = waveform[np.newaxis, ...]                                                                # shape [1, 5437]
        waveform = torch.from_numpy(waveform)												                # to torch [1, 5437]
        
        mel_specgram = torchaudio.transforms.MelSpectrogram(new_sample_rate)(waveform)                      # shape [1, 128, 331]
        mel_specgram = mel_specgram.detach().numpy()                                                        # to numpy to append
        note_array.append(mel_specgram)  
        index_array.append(index)                                                                           # put into array

for condition in dataset:               
    filename = os.listdir(condition)                                                                        # training data
    for i in range(250):
        print("processing (testfile) ... " + condition + " " + str(i))
        test_waveform, sample_rate = torchaudio.load(condition + '/' + filename[i])	                        # read waveform shape [2, 66150]
        new_sample_rate = sample_rate / RESAMPLE_RATE
        test_waveform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(test_waveform[0,:].view(1, -1))
        test_waveform = test_waveform.numpy()[0, :int(FRAME_SIZE)]

        test_waveform = test_waveform[np.newaxis, ...]                                          
        test_waveform = torch.from_numpy(test_waveform)
        
        mel_specgram = torchaudio.transforms.MelSpectrogram(new_sample_rate)(waveform)                      # shape [1, 128, 331]
        mel_specgram = mel_specgram.detach().numpy()                                                        # to numpy to append
        test_array.append(mel_specgram) 

os.chdir("../../")

tensor_note =  torch.Tensor(note_array).float()                                                             # all data to tenser
tensor_index = torch.Tensor(index_array).long()
tensor_test = torch.Tensor(test_array).float()

print(np.shape(tensor_note))

train_dataset = data.TensorDataset(tensor_note, tensor_index)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Sequential(
            nn.Linear(27648 , 1000),   # fully connected layer, output 10 classes
            nn.ReLU(),
            nn.Linear(1000, 4),  # fully connected layer, output 10 classes
            # nn.ReLU(),
            # nn.Linear(5000, 100),  # fully connected layer, output 10 classes
            # nn.ReLU(),
            # nn.Linear(100, 5)   # fully connected layer, output 10 classes
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization

# def set_seed(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
# set_seed(43)

cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

three_times = 0
jump = False

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        output, _ = cnn(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 5 == 0:
            test_output, _ = cnn(tensor_note)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == tensor_index.data.numpy()).astype(int).sum()) / float(tensor_index.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            if int(accuracy) == 1:
                three_times = three_times + 1
                print(three_times)
            else:
                three_times = 0
            if three_times == 4:
                jump = True
                break
    print("start save EPOCH : ", epoch)
    torch.save(cnn, "./model/harmonica_2d_ver3/harmonica_error_2d_model_" + str(epoch) + ".pth")
    torch.save(cnn.state_dict(), "./model/harmonica_2d_ver3/harmonica_error_2d_params_" + str(epoch) + ".pth")
    print("saved")
    if jump:
        break

print("training-----------done")

print("start save...")
torch.save(cnn, "./model/harmonica_2d_ver3/harmonica_error_2d_model.pth")
torch.save(cnn.state_dict(), "./model/harmonica_2d_ver3/harmonica_error_2d_params.pth")
print("saved")

test_output, _ = cnn(tensor_test)
print(test_output)
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print('[0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2] real number')

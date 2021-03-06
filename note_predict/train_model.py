import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import torchaudio
import matplotlib.pyplot as plt

# Hyper Parameters
BATCH_SIZE = 10  # num of training examples per minibatch
EPOCH = 15
LR = 0.001

index_array = np.load('../dataset/harmonica_note_test_dataset/index_array.npy')
r = ['Do', 'Re', 'Mi', 'Fa', 'So']
note_array = []
test_array = []
for note in r:                                                                              # training data
    for i in range(1, 31):
        waveform, sample_rate = torchaudio.load('../dataset/harmonica_note_test_dataset/harmonica_note_train/' + note + '_' + str(i) + '.wav', normalization=False)	# read waveform shape [2, 66150]
        new_sample_rate = sample_rate/4
        waveform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[0,:].view(1,-1))
        waveform = waveform.numpy()[0, :]													# shape [66150]

        waveform = waveform[np.newaxis, ...]                                                # shape [1, 66150]
        waveform = torch.from_numpy(waveform)												# to torch [1, 66150]

        waveform = waveform.detach().numpy()
        note_array.append(waveform)                                                         # put into array

for i in range(1, 14):
    test_waveform, sample_rate = torchaudio.load('../dataset/harmonica_note_test_dataset/harmonica_note_test/test_' + str(i) + '.wav')      # test data
    new_sample_rate = sample_rate/4
    test_waveform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(test_waveform[0,:].view(1, -1))
    test_waveform = test_waveform.numpy()[0, :]

    test_waveform = test_waveform[np.newaxis, ...]                                          
    test_waveform = torch.from_numpy(test_waveform)

    test_waveform = test_waveform.detach().numpy()
    test_array.append(test_waveform)

tensor_note =  torch.Tensor(note_array)                                             # all data to tenser
tensor_index = torch.Tensor(index_array).long()
tensor_test = torch.Tensor(test_array)

test_note = tensor_note
test_index = tensor_index

train_dataset = data.TensorDataset(tensor_note, tensor_index)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

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
            # nn.Tanh(),
            # nn.ReLU(),
            # nn.Linear(40000, 5000),  # fully connected layer, output 10 classes
            # nn.ReLU(),
            # nn.Linear(5000, 100),  # fully connected layer, output 10 classes
            # nn.ReLU(),
            # nn.Linear(100, 5)   # fully connected layer, output 10 classes
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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
            test_output, _ = cnn(test_note)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_index.data.numpy()).astype(int).sum()) / float(test_index.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            if accuracy == 1:
                three_times = three_times + 1
            else:
                three_times = 0
            if three_times == 4:
                jump = True
                break
    if jump:
        break

print("training-----------done")

print("start save...")
torch.save(cnn, "../model/harmonica_note_test_model/harmonica_model_5_2.pth")
torch.save(cnn.state_dict(), "../model/harmonica_note_test_model/harmonica_params_5_2.pth")
print("saved")

test_output, _ = cnn(tensor_test)
print(test_output)
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y + 1, 'prediction number')
print('[1 2 5 4 1 4 5 3 2 1 4 2 5] real number')
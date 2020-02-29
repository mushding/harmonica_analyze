import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import torchaudio
import matplotlib.pyplot as plt

# Hyper Parameters
BATCH_SIZE = 50  # num of training examples per minibatch
EPOCH = 10
LR = 0.001

index_array = np.load('../index_array.npy')
r = ['Do', 'Re', 'Mi', 'Fa', 'So']
note_array = []
test_array = []
for note in r:
    for i in range(1, 31):
        waveform, sample_rate = torchaudio.load('../wav/' + note + '_' + str(i) + '.wav')	# read waveform shape [2, 66150]
        waveform = waveform.numpy()[0, :]													# shape [66150]
        waveform = waveform[np.newaxis, ...]												# shape [1, 66150]
        waveform = torch.from_numpy(waveform)												# to torch [1, 66150]

        # plt.figure()
        # plt.plot(waveform)
        # plt.show()

        mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)          # shape [1, 128, 331]
        mel_specgram = mel_specgram.detach().numpy()                                        # to numpy to append
        note_array.append(mel_specgram)                                                     # shape [1, 1, 128, 331]
        print(np.shape(note_array))
        exit()

for i in range(1, 8):
    test_waveform, sample_rate = torchaudio.load('./test_wav/test_' + str(i) + '.wav')
    test_waveform = test_waveform.numpy()[0, :]
    test_waveform = test_waveform[np.newaxis, ...]
    test_waveform = torch.from_numpy(test_waveform)
    
    mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate)(test_waveform)
    mel_specgram = mel_specgram.detach().numpy()
    test_array.append(mel_specgram)

tensor_note =  torch.Tensor(note_array).float()
tensor_index = torch.Tensor(index_array).long()
tensor_test = torch.Tensor(test_array).float()

test_note = tensor_note
test_index = tensor_index

train_dataset = data.TensorDataset(tensor_note, tensor_index)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# plt.figure()
# plt.imshow(mel_specgram[0, ...])
# plt.show()
# exit()

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
            nn.Linear(83968, 5),   # fully connected layer, output 10 classes
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
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

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

test_output, _ = cnn(tensor_test)
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y + 1, 'prediction number')
print('[1 2 5 4 1 4 5] real number')
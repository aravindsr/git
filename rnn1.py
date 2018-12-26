# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 11:14:15 2018

@author: asesagiri
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)  # reproducibility


idx2char = ['h', 'i', 'e', 'l', 'o']

# Teach hihell -> ihello
x_data = [[0, 1, 0, 2, 3, 3]]   # hihell
y_data = [1, 0, 2, 3, 3, 4]    # ihello

# As we have one batch of samples, we will change them to variables only once
inputs = Variable(torch.LongTensor(x_data))
labels = Variable(torch.LongTensor(y_data))

num_classes = 5
input_size = 5
embedding_size = 1  # embedding size
hidden_size = 5  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 6  # |ihello| == 6
num_layers = 1  # one-layer rnn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(input_size=embedding_size,hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        h_0 = Variable(torch.zeros(num_layers, x.size(0), hidden_size))
        #print(h_0, num_layers, x.size(0), hidden_size)
        emb = self.embedding(x)
        #print(emb)
        emb = emb.view(batch_size, sequence_length, -1)
        #print(emb)
        # Propagate embedding through RNN
        # Input: (batch, seq_len, embedding_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)
        out, _ = self.rnn(emb, h_0)
        return self.fc(out.view(-1, num_classes))


# Instantiate RNN model
model = Model()
print(model)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Train the model
for epoch in range(20):
    outputs = model(inputs)
    #print(outputs)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(1)
    #print(outputs.max(1))
    idx = idx.data.numpy()
    #print(idx.squeeze())
    result_str = [idx2char[c] for c in idx.squeeze()]
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data))
    print("Predicted string: ", ''.join(result_str))

print("Learning finished!")
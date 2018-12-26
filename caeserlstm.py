# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 16:05:46 2018

@author: asesagiri
"""

key = 13
vocab = [char for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ-']


def encrypt(text):
    """Returns the encrypted form of 'text'."""
    indexes = [vocab.index(char) for char in text]
    encrypted_indexes = [(idx + key) % len(vocab) for idx in indexes]
    encrypted_chars = [vocab[idx] for idx in encrypted_indexes]
    encrypted = ''.join(encrypted_chars)
    return encrypted

# -----------------------------------------------------------------------------
assert encrypt('ABCDEFGHIJKLMNOPQRSTUVWXYZ-') == 'NOPQRSTUVWXYZ-ABCDEFGHIJKLM'
# -----------------------------------------------------------------------------

import random
import torch

num_examples = 20
message_length = 5


def dataset(num_examples):
    """Returns a list of 'num_examples' pairs of the form (encrypted, original).
    
    Both elements of the pair are tensors containing indexes of each character
    of the corresponding encrypted or original message.
    """
    dataset = []
    for x in range(num_examples):
        ex_out = ''.join([random.choice(vocab) for x in range(message_length)])
        # may be: MANR-TQNNAFEGIDE-OXQZANSVEMJXWSU
        ex_in = encrypt(''.join(ex_out))
        # may be: ZN-DMFC--NSRTVQRMAJCLN-EHRZWJIEG
        ex_in = [vocab.index(x) for x in ex_in]
        # may be: [25, 13, 26, 3, 12, 5, 2, 26, 26, ...
        ex_out = [vocab.index(x) for x in ex_out]
        # may be: [12, 0, 13, 17, 26, 19, 16, 13, ...
        dataset.append([torch.tensor(ex_in), torch.tensor(ex_out)])
    return dataset

# -----------------------------------------------------------------------------
# print(dataset(1))
# -----------------------------------------------------------------------------

embedding_dim = 5
hidden_dim = 10
vocab_size = len(vocab)

embed = torch.nn.Embedding(vocab_size, embedding_dim)
lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
linear = torch.nn.Linear(hidden_dim, vocab_size)
softmax = torch.nn.functional.softmax
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(embed.parameters()) + list(lstm.parameters())
                            + list(linear.parameters()), lr=0.001)

# -----------------------------------------------------------------------------

def zero_hidden():
    return (torch.zeros(1, 1, hidden_dim),
            torch.zeros(1, 1, hidden_dim))


num_epochs = 100

accuracies, max_accuracy = [], 0
for x in range(num_epochs):
    print('Epoch: {}'.format(x))
    for encrypted, original in dataset(num_examples):
        # encrypted.size() = [64]
        
        lstm_in = embed(encrypted)
        #print(lstm_in)
        # lstm_in.size() = [64, 5]. This is a 2D tensor, but LSTM expects 
        # a 3D tensor. So we insert a fake dimension.
        lstm_in = lstm_in.unsqueeze(1)
        #print(lstm_in)
        # lstm_in.size() = [64, 1, 5]
        # Get outputs from the LSTM.
        lstm_out, lstm_hidden = lstm(lstm_in, zero_hidden())
        # lstm_out.size() = [64, 1, 10]
        # Apply the affine transform.
        scores = linear(lstm_out)
        # scores.size() = [64, 1, 27], but loss_fn expects a tensor
        # of size [64, 27, 1]. So we switch the second and third dimensions.
        scores = scores.transpose(1, 2)
        # original.size() = [64], but original should also be a 2D tensor
        # of size [64, 1]. So we insert a fake dimension.
        original = original.unsqueeze(1)
        # Calculate loss.
        loss = loss_fn(scores, original) 
        # Backpropagate
        loss.backward()
        # Update weights
        optimizer.step()
    print('Loss: {:6.4f}'.format(loss.item()))




    with torch.no_grad():
        matches, total = 0, 0
        for encrypted, original in dataset(num_examples):
            lstm_in = embed(encrypted)
            lstm_in = lstm_in.unsqueeze(1)
            lstm_out, lstm_hidden = lstm(lstm_in, zero_hidden())
            scores = linear(lstm_out)
            # Compute a softmax over the outputs
            predictions = softmax(scores, dim=2)
            # Choose the letter with the maximum probability
            _, batch_out = predictions.max(dim=2)
            # Remove fake dimension
            batch_out = batch_out.squeeze(1)
            # Calculate accuracy
            #print("original is",[vocab[x] for x in original])
            #print("predicted is",[vocab[x] for x in batch_out])
            matches += torch.eq(batch_out, original).sum().item()
            total += torch.numel(batch_out)
        accuracy = matches / total
        print('Accuracy: {:4.2f}%'.format(accuracy * 100))

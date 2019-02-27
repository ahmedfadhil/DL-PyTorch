import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# Step1: load dataset
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

print(train_dataset.train_data.size())
print(train_dataset.train_labels.size())
print(test_dataset.test_data.size())
print(test_dataset.test_labels.size())

# Make dataset iterable
batch_size = 100
n_iter = 3000
n_epochs = n_iter / (len(train_dataset) / batch_size)
n_epochs = int(n_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Build the mode
class RnnModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RnnModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Num of hidden layers
        self.hidden_layer = layer_dim

        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        h0 = Variable(torch.zeros(self.layer_dim, input.size(0), self.hidden_dim))
        output, hn = self.rnn(input, h0)
        output = self.fc(output[:, -1:])
        return output


#     Instantiate model class
input_dim = 28
hidden_dim = 100
layer_dim = 1
output_dim = 10

model = RnnModel(input_dim, hidden_dim, layer_dim, output_dim)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Parameter in-depth
len(list(model.parameters()))

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import dlc_practical_prologue as prologue

pairs = prologue.generate_pair_sets(1000)
train_input=pairs[0]    #Nx2x14x14
train_target=pairs[1]   #N
train_classes=pairs[2]  #Nx2
test_input=pairs[3]     #Nx2x14x14
test_target=pairs[4]    #N
test_classes=pairs[5]   #Nx2

################################################################################

class ConvNet(nn.Module):
    '''
    Adapted from Week 4
    Input size :            2x14x14
    nn.Conv2d(2, 32, k=3) : 32x12x12
    F.max_pool2d(k=2) :     32x6x6
    nn.Conv2d(32, 64, k=3): 64x4x4
    F.max_pool2d(k=2):      64x2x2
    '''
    def __init__(self, nb_hidden=100):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

################################################################################

def train_model(model, train_input, train_target):
    #Taken from Week 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-1)
    nb_epochs = 250
    mini_batch_size= 100

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()

################################################################################

def compute_nb_errors(model, data_input, data_target):
    #Taken from week 5
    nb_data_errors = 0
    mini_batch_size= 100
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

################################################################################

train_E=[]
test_E=[]

for k in range(5):
    model=ConvNet(100)
    train_model(model,train_input,train_target)
    train_E.append(compute_nb_errors(model, train_input, train_target) / train_input.size(0) * 100)
    test_E.append(compute_nb_errors(model, test_input, test_target) / test_input.size(0) * 100)
    print('train_error {:.02f}% test_error {:.02f}%'.format(test_E[-1],train_E[-1]))

train_E=torch.tensor(train_E)
test_E=torch.tensor(train_E)

print(f'Train error average : {train_E.mean()}')
print(f'Train error std : {train_E.std()}')
print(f'Test error average : {test_E.mean()}')        
print(f'Test error std : {test_E.std()}')
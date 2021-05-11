import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    '''
    Simple MLP layer.
    Input size  :            2x14x14
    Vector size :            392 
    The pictures are vectorized and then concatenated.
    '''
    def __init__(self, nb_hidden=100):
        super().__init__()
        self.fc1 = nn.Linear(392, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FullyConvNet(nn.Module):
    '''
    Input size :                2x14x14
    nn.Conv2d(2, 32, k=3) :     32x12x12
    F.max_pool2d(k=2) :         32x6x6
    nn.Conv2d(32, 64, k=3):     64x4x4
    F.max_pool2d(k=2):          64x2x2
    nn.Conv2d(64, N, k=(2,2)):  Nx1x1
    nn.Conv2d(N, 2, k=(2,2)):   2x1x1
    '''
    def __init__(self, nb_hidden=32):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, nb_hidden, kernel_size=2)
        self.conv4 = nn.Conv2d(nb_hidden, 2, kernel_size=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.view(-1,x.size(1)) #Returns a vector with the channels concatenated 
        return x

class ConvNet(nn.Module):
    '''
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

class ConvNetAux(nn.Module):
    '''
    Input size :            2x14x14
    nn.Conv2d(2, 32, k=3) : 32x12x12
    F.max_pool2d(k=2) :     32x6x6
    nn.Conv2d(32, 64, k=3): 64x4x4
    F.max_pool2d(k=2):      64x2x2
    '''
    def __init__(self, nb_digits, nb_hidden=100, p=0.2):
        super().__init__()        
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3)
        self.dropout1 = nn.Dropout2d(p = 0.2) # Removes entire channel randomly, so dropout -> maxpool = maxpool -> dropout
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3)
        self.dropout2 = nn.Dropout2d(p = 0.2) # Removes entire channel randomly, so dropout -> maxpool = maxpool -> dropout
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        self.fc1 = nn.Linear(128, nb_digits)
        self.fc2 = nn.Linear(2 * nb_digits, 2)
        
    def forward(self, x):
        batch_size, nb_channels, image_rows, image_cols = x.size()
        
        #Concatenate the channels along the batch dimension. 
        x = x.view(nb_channels * batch_size, 1, image_rows, image_cols)
        
        # Images of both channels are trained indifferently
        x = F.relu(self.maxpool1(self.dropout1(self.conv1(x))))
        x = F.relu(self.maxpool2(self.dropout2(self.conv2(x))))
        
        # Vectorize and concatenate pictures for FC layer. 
        # Pictures of channel 2 are concatenated at the end of pictures from channel 1.
        # Each digit is predicted separately 
        digits_pred = self.fc1(x.view(2 * batch_size, -1))
        
        x = F.softmax(digits_pred, dim = 1)
        
        # Each predicted digit is concatenated with the digit to which it must be compared
        x = x.view(batch_size, -1)
        
        x = self.fc2(x)
        return x, digits_pred
import torch
from torch import nn
from torch.nn import functional as F



class MLP(nn.Module):
    """
    Args: 
        p::float
            Probability of dropping channel
        nb_hidden::int
            Number of hidden units in first FC layer
    Return:
        x::Tensor[batch_size,2]
            Prediction of the comparison
    """

    def __init__(self, p, nb_hidden1, nb_hidden2):
        super().__init__()
        self.fc1 = nn.Linear(392, nb_hidden1)
        self.dropout1 = nn.Dropout(p = p)
        self.fc2 = nn.Linear(nb_hidden1, nb_hidden2)
        self.dropout2 = nn.Dropout(p = p)
        self.fc3 = nn.Linear(nb_hidden2, 2)

    def forward(self, x):
        x = x.view(-1, 392)
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = self.fc3(x)
        return x


class MLPAux(nn.Module):
    """
    Args: 
        p::float
            Probability of dropping channel
        nb_hidden::int
            Number of hidden units in first FC layer
    Return:
        x::Tensor[batch_size,2]
            Prediction of the comparison
        digits_pred::Tensor[2*batch_size,10]
            Prediction of the digit of each picture
    """

    # MLP with dropout and auxiliary loss.
    # Input size  :            2x14x14
    # Vector size :            392 



    def __init__(self, p, nb_hidden):
        super().__init__()
        self.fc1 = nn.Linear(196, nb_hidden)
        self.dropout1 = nn.Dropout(p=p)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.dropout2 = nn.Dropout(p=p)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        batch_size, nb_channels, image_rows, image_cols = x.size()

        # Vectorize pictures and concatenates the pairs of pictures.
        # Vectors are of size 14*14*2
        x = x.view(nb_channels * batch_size, image_rows*image_cols)
        x = F.relu(self.dropout1(self.fc1(x)))

        # Separate pairs of digits to predict each digit separately
        digits_pred = F.relu(self.fc2(x.view(2 * batch_size, -1)))

        x = F.softmax(digits_pred, dim=1)
        
        # Each predicted digit is concatenated with the digit to which it must be compared.
        x = x.view(batch_size, -1)
        x = self.fc3(x)
        return x, digits_pred


class ConvNet(nn.Module):
    """
    Args: 
        p::float
            Probability of dropping channel
        nb_hidden::int
            Number of hidden units in first FC layer
    Return:
        x::Tensor[batch_size,2]
            Prediction of the comparison
    """
    # ConvNet with dropout
    # Input size :                2x14x14
    # nn.Conv2d(2, 32, k=3) : 32x12x12
    # F.max_pool2d(k=2) :     32x6x6
    # nn.Conv2d(32, 64, k=3): 64x4x4
    # F.max_pool2d(k=2):      64x2x2

    def __init__(self, p, nb_hidden, ksize, padding, n_chan_1, n_chan_2):
        super().__init__()
        
        import math
        
        if padding == 100:
            padding = 0
        elif padding == 101:
            padding = (ksize - 1) / 2
            if padding % 1:
                padding = (math.floor(padding), math.ceil(padding))
            elif type(padding) == float:
                padding = int(padding)
        
        
        self.conv1 = nn.Conv2d(2, n_chan_1, kernel_size=ksize, padding = padding)
        self.dropout1 = nn.Dropout2d(p=p)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(n_chan_1, n_chan_2, kernel_size=ksize, padding = padding)
        self.dropout2 = nn.Dropout2d(p=p)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Empirically determining length of flattened tensor.
        with torch.no_grad():
            stub = torch.empty((1, 2, 14, 14))
            stub = self.maxpool2(self.conv2(self.maxpool1(self.conv1(stub))))
            self.size = len(stub.view(-1))
            print(self.size)
        
        self.fc1 = nn.Linear(self.size, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.relu(self.maxpool1(self.dropout1(self.conv1(x))))
        x = F.relu(self.maxpool2(self.dropout2(self.conv2(x))))
        x = F.relu(self.fc1(x.view(-1, self.size)))
        x = self.fc2(x)
        return x


class ConvNetAux(nn.Module):
    """
    Args: 
        p::float
            Probability of dropping channel
        nb_hidden::int
            Number of hidden units in first FC layer
    Return:
        x::Tensor[batch_size,2]
            Prediction of the comparison
        digits_pred::Tensor[2*batch_size,10]
            Prediction of the digit of each picture
    """

    def __init__(self, p , n_chan_1, n_chan_2, ksize, padding):
        super().__init__()
        
        import math
        
        if padding == 100:
            padding = 0
        elif padding == 101:
            padding = (ksize - 1) / 2
            if padding % 1:
                padding = (math.floor(padding), math.ceil(padding))
            elif type(padding) == float:
                padding = int(padding)
        
        
        
        self.conv1 = nn.Conv2d(1, n_chan_1, kernel_size=ksize, padding = padding)
        self.dropout1 = nn.Dropout2d(p=p)  
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(n_chan_1, n_chan_2, kernel_size=ksize, padding = padding)
        self.dropout2 = nn.Dropout2d(p=p)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        with torch.no_grad():
            stub = torch.empty((1, 1, 14, 14))
            stub = self.maxpool2(self.conv2(self.maxpool1(self.conv1(stub))))
            self.size = len(stub.view(-1))
            print(self.size)
        
        self.fc1 = nn.Linear(self.size, 10)
        
        self.fc2 = nn.Linear(2 * 10, 2)

    def forward(self, x):
        batch_size, nb_channels, image_rows, image_cols = x.size()

        # Concatenate the channels along the batch dimension. Order is : [P1 C1, P1 C2, P2 C1, P2 C2, ...]
        x = x.view(nb_channels * batch_size, 1, image_rows, image_cols)

        # Images of both channels are trained indifferently
        x = F.relu(self.maxpool1(self.dropout1(self.conv1(x))))
        x = F.relu(self.maxpool2(self.dropout2(self.conv2(x))))

        # Vectorize pictures for FC layer.
        # Each digit is predicted separately
        digits_pred = self.fc1(x.view(2 * batch_size, -1))

        x = F.softmax(digits_pred, dim=1)
        
        # Each predicted digit is concatenated with the digit to which it must be compared
        x = x.view(batch_size, -1)

        x = self.fc2(x)
        return x, digits_pred


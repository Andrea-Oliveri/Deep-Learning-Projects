import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    """
    Multilayer perceptron with two hidden layers.
        Predicts for each pair of digit if the first digit is lesser or equal to the second.
    Input:
        input::[torch.Tensor]
            Tensor of shape (batch_size,2,14,14) which contains the pair of 14x14 digits.
    Args: 
        p::[float]
            Probability of an element to be zeroed in the second and third layer.
        nb_hidden1::[int]
            Number of hidden units in second layer.
        nb_hidden2::[int]
            Number of hidden units in third layer.    
    Returns:
        x::[torch.Tensor]
            Tensor of shape (batch_size, 2) containing the comparison prediction.
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
    Multilayer perceptron with two hidden layers.
        Predicts for each picture the corresponding digit.
        Predicts for each pair of digit if the first digit is lesser or equal to the second.
    
    Input:
        input::[torch.Tensor]
            Tensor of shape (batch_size,2,14,14) which contains the pair of 14x14 digits.
    Args: 
        p::[float]
            Probability of an element to be zeroed in the second and third layer.
        nb_hidden::[int]
            Number of hidden units in second layer.
    Returns:
        x::[torch.Tensor]
            Tensor of shape (batch_size, 2) containing the comparison prediction.
        digits_pred::[torch.Tensor]
            Tensor of shape (batch_size, 10) containing the digit prediction.
    """

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
        x = x.view(nb_channels * batch_size, image_rows*image_cols)
        x = F.relu(self.dropout1(self.fc1(x)))
        # Each digit is predicted separately
        digits_pred = F.relu(self.fc2(x.view(2 * batch_size, -1)))
        x = F.softmax(digits_pred, dim=1)
        # Each predicted digit is concatenated with the digit to which it must be compared.
        x = x.view(batch_size, -1)
        x = self.fc3(x)

        return x, digits_pred


class ConvNet(nn.Module):
    """
    Convolutional neural network with two convolutional layers and an MLP classifier.
        Predicts for each pair of digit if the first digit is lesser or equal to the second.
    
    Input:
        input::[torch.Tensor]
            Tensor of shape (batch_size,2,14,14) which contains the pair of 14x14 digits.
    Args: 
        p::[float]
            Probability of a channel to be zeroed after the first and second convolutional layers.
        nb_hidden::[int]
            Number of hidden units in the MLP.
        k_size::[int]
            Kernel size of the convolutional layers.
        padding::[int]
            Specifies the size of a zeroed frame added around the input.
            If padding is set to 1
        n_channel1::[int]
            Number of output channels in first convolutional layer.
        n_channel2::[int]
            Number of output channels in second convolutional layer.    
    Returns:
        x::[torch.Tensor]
            Tensor of shape (batch_size, 2) containing the comparison prediction.
    """

    def __init__(self, p, nb_hidden, k_size, padding, nb_channel1, nb_channel2):
        super().__init__()
        
        import math
        
        if padding == 100:
            padding = 0
        elif padding == 101:
            padding = (k_size - 1) / 2
            if padding % 1:
                padding = (math.floor(padding), math.ceil(padding))
            elif type(padding) == float:
                padding = int(padding)
        
        
        self.conv1 = nn.Conv2d(2, nb_channel1, kernel_size=k_size, padding = padding)
        self.dropout1 = nn.Dropout2d(p=p)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(nb_channel1, nb_channel2, kernel_size=k_size, padding = padding)
        self.dropout2 = nn.Dropout2d(p=p)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Empirically determining length of flattened tensor.
        with torch.no_grad():
            stub = torch.empty((1, 2, 14, 14))
            stub = self.maxpool2(self.conv2(self.maxpool1(self.conv1(stub))))
            self.size = len(stub.view(-1))
        
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
    Convolutional neural network with two convolutional layers and an MLP classifier.
        Predicts for each picture the corresponding digit.
        Predicts for each pair of digit if the first digit is lesser or equal to the second.
    
    Input:
        input::[torch.Tensor]
            Tensor of shape (batch_size,2,14,14) which contains the pair of 14x14 digits.
    Args: 
        p::[float]
            Probability of a channel to be zeroed after the first and second convolutional layers.
        nb_hidden::[int]
            Number of hidden units in the MLP.
        k_size::[int]
            Kernel size of the convolutional layers.
        padding::[int]
            Specifies the size of a zeroed frame added around the input.
            If padding is set to 1
        nb_channel1::[int]
            Number of output channels in first convolutional layer.
        nb_channel2::[int]
            Number of output channels in second convolutional layer.    
    Returns:
        x::[torch.Tensor]
            Tensor of shape (batch_size, 2) containing the comparison prediction.
        digits_pred::[torch.Tensor]
            Tensor of shape (batch_size, 10) containing the digit prediction.
    """

    def __init__(self, p, k_size, padding, nb_channel1, nb_channel2):
        super().__init__()
        
        import math
        
        if padding:
            padding = (k_size - 1) / 2
            if padding % 1:
                padding = (math.floor(padding), math.ceil(padding))
            elif type(padding) == float:
                padding = int(padding)        
        
        self.conv1 = nn.Conv2d(1, nb_channel1, kernel_size=k_size, padding = padding)
        self.dropout1 = nn.Dropout2d(p=p)  
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(nb_channel1, nb_channel2, kernel_size=k_size, padding = padding)
        self.dropout2 = nn.Dropout2d(p=p)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        with torch.no_grad():
            stub = torch.empty((1, 1, 14, 14))
            stub = self.maxpool2(self.conv2(self.maxpool1(self.conv1(stub))))
            self.size = len(stub.view(-1))
        
        self.fc1 = nn.Linear(self.size, 10)
        self.fc2 = nn.Linear(2 * 10, 2)

    def forward(self, x):
        batch_size, nb_channels, image_rows, image_cols = x.size()

        # Concatenate the channels along the batch dimension. Order is : [P1 C1, P1 C2, P2 C1, P2 C2, ...]
        x = x.view(nb_channels * batch_size, 1, image_rows, image_cols)

        # Images of both channels are trained indifferently
        x = F.relu(self.maxpool1(self.dropout1(self.conv1(x))))
        x = F.relu(self.maxpool2(self.dropout2(self.conv2(x))))

        # Each digit is predicted separately
        digits_pred = self.fc1(x.view(2 * batch_size, -1))

        x = F.softmax(digits_pred, dim=1)
        # Each predicted digit is concatenated with the digit to which it must be compared
        x = x.view(batch_size, -1)
        x = self.fc2(x)

        return x, digits_pred


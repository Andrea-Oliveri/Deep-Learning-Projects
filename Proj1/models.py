from torch import nn
from torch.nn import functional as F

class FullyConnectedNet(nn.Module):
    """
    Fully Connected Network with two hidden layers.
        Predicts for each pair of digit if the first digit is lesser or equal to the second.
    Input:
        input::[torch.Tensor]
            Tensor of shape (batch_size, 2, 14, 14) which contains the pair of 14x14 digits.
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
        self.flattened_size = 2*14*14
        self.fc1 = nn.Linear(self.flattened_size, nb_hidden1)
        self.dropout1 = nn.Dropout(p = p)
        self.fc2 = nn.Linear(nb_hidden1, nb_hidden2)
        self.dropout2 = nn.Dropout(p = p)
        self.fc3 = nn.Linear(nb_hidden2, 2)
        

    def forward(self, x):
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = self.fc3(x)
        return x


class FullyConnectedNetAux(nn.Module):
    """
    Fully Connected Network with two hidden layers.
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
        self.flattened_size = 1*14*14
        self.fc1 = nn.Linear(self.flattened_size, nb_hidden)
        self.dropout1 = nn.Dropout(p = p)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.dropout2 = nn.Dropout(p = p)
        self.fc3 = nn.Linear(20, 2)
        

    def forward(self, x):
        batch_size, nb_channels, image_rows, image_cols = x.size()

        # Each digit is predicted separately
        x = x.view(nb_channels * batch_size, image_rows * image_cols)
        x = self.dropout1(self.fc1(x))
        
        digits_pred = self.fc2(x)
        x = digits_pred

        # Each predicted digit is concatenated with the digit to which it must be compared.
        x = x.view(batch_size, -1)
        x = self.fc3(x)

        return x, digits_pred


class ConvolutionalNet(nn.Module):
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
            Specifies how much padding to use with each convolutional layer.
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
        
        self.conv1 = nn.Conv2d(2, nb_channel1, kernel_size=k_size, padding = padding)
        self.dropout1 = nn.Dropout2d(p = p)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(nb_channel1, nb_channel2, kernel_size = k_size, padding = padding)
        self.dropout2 = nn.Dropout2d(p = p)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        
        # Calculating length of flattened tensor.
        size_input = 14
        size_after_conv1    = (size_input - k_size + 2 * padding) + 1
        size_after_maxpool1 = size_after_conv1 // 2
        size_after_conv2    = (size_after_maxpool1 - k_size + 2 * padding) + 1
        size_after_maxpool2 = size_after_conv2 // 2
        self.flattened_size = (size_after_maxpool2 ** 2) * nb_channel2
        
        self.fc1 = nn.Linear(self.flattened_size, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)
        

    def forward(self, x):
        x = F.relu(self.maxpool1(self.dropout1(self.conv1(x))))
        x = F.relu(self.maxpool2(self.dropout2(self.conv2(x))))
        x = F.relu(self.fc1(x.view(-1, self.flattened_size)))
        x = self.fc2(x)
        return x


class ConvolutionalNetAux(nn.Module):
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
            Specifies how much padding to use with each convolutional layer.
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
        
        self.conv1 = nn.Conv2d(1, nb_channel1, kernel_size = k_size, padding = padding)
        self.dropout1 = nn.Dropout2d(p = p)  
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(nb_channel1, nb_channel2, kernel_size = k_size, padding = padding)
        self.dropout2 = nn.Dropout2d(p = p)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        
        # Calculating length of flattened tensor.
        size_input = 14
        size_after_conv1    = (size_input - k_size + 2 * padding) + 1
        size_after_maxpool1 = size_after_conv1 // 2
        size_after_conv2    = (size_after_maxpool1 - k_size + 2 * padding) + 1
        size_after_maxpool2 = size_after_conv2 // 2
        self.flattened_size = (size_after_maxpool2 ** 2) * nb_channel2
        
        self.fc1 = nn.Linear(self.flattened_size, 10)
        self.fc2 = nn.Linear(2 * 10, 2)
        
        
    def forward(self, x):
        batch_size, nb_channels, image_rows, image_cols = x.size()

        # Concatenate the channels along the batch dimension. Order is : [P1 C1, P1 C2, P2 C1, P2 C2, ...]
        x = x.view(nb_channels * batch_size, 1, image_rows, image_cols)

        # Each digit is predicted separately
        x = F.relu(self.maxpool1(self.dropout1(self.conv1(x))))
        x = F.relu(self.maxpool2(self.dropout2(self.conv2(x))))

        digits_pred = self.fc1(x)

        x = digits_pred
        # Each predicted digit is concatenated with the digit to which it must be compared
        x = x.view(batch_size, -1)
        x = self.fc2(x)

        return x, digits_pred


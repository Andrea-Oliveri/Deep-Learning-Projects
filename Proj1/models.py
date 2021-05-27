from torch import nn
from torch.nn import functional as F


class FullyConnectedNet(nn.Module):
    """Fully Connected Network which takes inputs containing an image per
    channel and just flattenes all channels into one single long vector
    and processes it to predict if the first channel digit is larger than 
    the second channel one."""
    
    def __init__(self, p, nb_hidden1, nb_hidden2, nb_hidden3):
        """Constructor of FullyConnectedNet class, which instanciates the
        layers composing this net using the desired number of hidden
        units of each layer, and probability of dropout. 
        
        Args:
            p::[float]
                Dropout probability of all Dropot layers.
            nb_hidden1::[int]
                Number of units in first hidden layer.
            nb_hidden2::[int]
                Number of units in second hidden layer.    
            nb_hidden3::[int]
                Number of units in third hidden layer.    
        """
        super().__init__()
        
        self.flattened_size = 2*14*14
        self.fc1 = nn.Linear(self.flattened_size, nb_hidden1)
        self.dropout1 = nn.Dropout(p = p)
        self.fc2 = nn.Linear(nb_hidden1, nb_hidden2)
        self.dropout2 = nn.Dropout(p = p)
        self.fc3 = nn.Linear(nb_hidden2, nb_hidden3)
        self.dropout3 = nn.Dropout(p = p)
        self.fc4 = nn.Linear(nb_hidden3, 2)
        

    def forward(self, inputs):
        """Method performing the forward pass of the net using inputs.
        
        Args:
            inputs::[torch.tensor]
                Input tensor of shape (batch_size, 2, 14, 14) in which each
                element of the batch is an image with two channels, where 
                each channel is an mnist digit.
        Returns:
            out::[torch.tensor]
                The output tensor obtained by performing the forward pass on x,
                of shape (batch_size, 2). Predicts whether the digit in the 
                first channel of the input is larger than that in the second 
                channel.    
        """
        x = inputs.view(-1, self.flattened_size)
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = F.relu(self.dropout3(self.fc3(x)))
        out = self.fc4(x)
        return out



class FullyConnectedNetAux(nn.Module):
    """Fully Connected Network which takes inputs containing an image per
    channel and reuses the same weights to first make a prediction of the 
    digit contained by each channel, and finally uses these predictions to
    determine if the first channel digit is larger than the second channel
    one."""

    def __init__(self, p, nb_hidden1, nb_hidden2):
        """Constructor of FullyConnectedNetAux class, which instanciates the
        layers composing this net using the desired number of hidden
        units of each layer, and probability of dropout. 
        
        Args:
            p::[float]
                Dropout probability of all Dropot layers.
            nb_hidden1::[int]
                Number of units in first hidden layer.
            nb_hidden2::[int]
                Number of units in second hidden layer.    
        """
        super().__init__()
        
        self.flattened_size = 1*14*14
        self.fc1 = nn.Linear(self.flattened_size, nb_hidden1)
        self.dropout1 = nn.Dropout(p = p)
        self.fc2 = nn.Linear(nb_hidden1, 10)
        self.fc3 = nn.Linear(2 * 10, nb_hidden2)
        self.dropout2 = nn.Dropout(p = p)
        self.fc4 = nn.Linear(nb_hidden2, 2)
        

    def forward(self, inputs):
        """Method performing the forward pass of the net using inputs.
        
        Args:
            inputs::[torch.tensor]
                Input tensor of shape (batch_size, 2, 14, 14) in which each
                element of the batch is an image with two channels, where 
                each channel is an mnist digit.
        Returns:
            out::[torch.tensor]
                The output tensor obtained by performing the forward pass on x,
                of shape (batch_size, 2). Predicts whether the digit in the 
                first channel of the input is larger than that in the second 
                channel.
            digits_pred::[torch.tensor]
                Tensor of shape (2 * batch_size, 10) containing the digits
                predicted for each channel of each image in the input tensor.
        """
        batch_size, nb_channels, image_rows, image_cols = inputs.size()

        # Each digit is predicted separately
        x = inputs.view(nb_channels * batch_size, image_rows * image_cols)
        x = F.relu(self.dropout1(self.fc1(x)))
        
        digits_pred = self.fc2(x)
        x = digits_pred

        # Each predicted digit is concatenated with the digit to which it
        # must be compared.
        x = x.view(batch_size, -1)
        x = F.relu(self.dropout2(self.fc3(x)))
        out = self.fc4(x)
        
        return out, digits_pred



class ConvolutionalNet(nn.Module):
    """Convolutional Network which takes inputs containing an image per
    channel and merges the channels with its first convolutional filter, 
    then keeps on processing the result to predict if the first channel digit
    is larger than the second channel one."""

    def __init__(self, p, k_size, padding, nb_channel1, nb_channel2, 
                 nb_hidden1, nb_hidden2):
        """Constructor of ConvolutionalNet class, which instanciates the
        layers composing this net using the desired number channels, padding
        and kernel size for the convolutional layers, number of hidden
        units of the linear layers and probability of dropout. 
        
        Args:
            p::[float]
                Dropout probability of all Dropot layers.
            k_size::[int]
                Kernel size of convolutional layers.
            padding::[int]
                Amount of zero-padding to use in convolutional layers.
            nb_channel1::[int]
                Number of channels outputed by first convolutional layer. 
            nb_channel2::[int]
                Number of channels outputed by second convolutional layer. 
            nb_hidden1::[int]
                Number of units in first linear layer.
            nb_hidden2::[int]
                Number of units in second linear layer.    
        """
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
        
        self.fc1 = nn.Linear(self.flattened_size, nb_hidden1)
        self.fc2 = nn.Linear(nb_hidden1, nb_hidden2)
        self.fc3 = nn.Linear(nb_hidden2, 2)
        

    def forward(self, inputs):
        """Method performing the forward pass of the net using inputs.
        
        Args:
            inputs::[torch.tensor]
                Input tensor of shape (batch_size, 2, 14, 14) in which each
                element of the batch is an image with two channels, where 
                each channel is an mnist digit.
        Returns:
            out::[torch.tensor]
                The output tensor obtained by performing the forward pass on x,
                of shape (batch_size, 2). Predicts whether the digit in the 
                first channel of the input is larger than that in the second 
                channel.    
        """
        x = F.relu(self.maxpool1(self.dropout1(self.conv1(inputs))))
        x = F.relu(self.maxpool2(self.dropout2(self.conv2(x))))
        x = F.relu(self.fc1(x.view(-1, self.flattened_size)))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        
        return out



class ConvolutionalNetAux(nn.Module):
    """Convolutional Network which takes inputs containing an image per
    channel and reuses the same weights to first make a prediction of the 
    digit contained by each channel, and finally uses these predictions to
    determine if the first channel digit is larger than the second channel
    one."""
    
    def __init__(self, p, k_size, padding, nb_channel1, nb_channel2, nb_hidden):
        """Constructor of ConvolutionalNet class, which instanciates the
        layers composing this net using the desired number channels, padding
        and kernel size for the convolutional layers, number of hidden
        units of the linear layer and probability of dropout. 
        
        Args:
            p::[float]
                Dropout probability of all Dropot layers.
            k_size::[int]
                Kernel size of convolutional layers.
            padding::[int]
                Amount of zero-padding to use in convolutional layers.
            nb_channel1::[int]
                Number of channels outputed by first convolutional layer. 
            nb_channel2::[int]
                Number of channels outputed by second convolutional layer. 
            nb_hidden::[int]
                Number of units in linear layer.
        """
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
        self.fc2 = nn.Linear(2 * 10, nb_hidden)
        self.fc3 = nn.Linear(nb_hidden, 2)
        
        
    def forward(self, inputs):
        """Method performing the forward pass of the net using inputs.
        
        Args:
            inputs::[torch.tensor]
                Input tensor of shape (batch_size, 2, 14, 14) in which each
                element of the batch is an image with two channels, where 
                each channel is an mnist digit.
        Returns:
            out::[torch.tensor]
                The output tensor obtained by performing the forward pass on x,
                of shape (batch_size, 2). Predicts whether the digit in the 
                first channel of the input is larger than that in the second 
                channel.
            digits_pred::[torch.tensor]
                Tensor of shape (2 * batch_size, 10) containing the digits
                predicted for each channel of each image in the input tensor.
        """
        batch_size, nb_channels, image_rows, image_cols = inputs.size()

        # Concatenate the channels along the batch dimension. 
        # Order is : [P1 C1, P1 C2, P2 C1, P2 C2, ...]
        x = inputs.view(nb_channels * batch_size, 1, image_rows, image_cols)

        # Each digit is predicted separately
        x = F.relu(self.maxpool1(self.dropout1(self.conv1(x))))
        x = F.relu(self.maxpool2(self.dropout2(self.conv2(x))))

        digits_pred = self.fc1(x.view(nb_channels * batch_size, -1))

        x = digits_pred
        # Each predicted digit is concatenated with the digit to which it
        # must be compared.
        x = x.view(batch_size, -1)
        x = F.relu(self.fc2(x))
        out = self.fc3(x)

        return out, digits_pred


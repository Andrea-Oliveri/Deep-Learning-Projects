# -*- coding: utf-8 -*-

import math
from torch import empty


class Initializer(object):
    """Abstract Class from which different weight and bias initializers will
    inherit from."""
    
    def __init__(self):
        """Stub constructor of Initializer. Raises an AssertionError as the 
        abstract class Initializer can't be instanciated."""
        assert type(self) != Initializer, \
               "Abstract Class Initializer can't be instanciated."
    
    def __call__(self, shape):
        """Special method defined for conveniently calling the _generate method.
        Args:
            shape::[tuple]
                See _generate for the explaination of the shape parameter.
        Returns:
            initialized_tensor::[torch.Tensor]
                See _generate for the explaination of the returned value
                initialized_tensor.
        """
        return self._generate(shape)
    
    def _generate(self, shape):
        """Stub function. Child classes will override it such that it will
        generate a tensor of desired shape initialized with a technique
        different for each child class.
        Args:
            shape::[tuple]
                The shape of the tensor we want to create and initialize.
        Returns:
            initialized_tensor::[torch.Tensor]
                The tensor created and initialized with a technique different 
                for each child class.
        """
        return
    
    def _fan_in_fan_out_from_shape(self, shape):
        """Utilitary method which checks the shape parameter has length 2
        (shape describes a matrix) and returns the fan_in and fan_out values
        of the Linear layer having a weight of this shape.
        Args:
            shape::[tuple]
                The shape of the weight matrix.
        Returns:
            fan_in::[int]
                The number of units in the Linear layer preceding the one
                having weight matrix with shape as in the parameter.
            fan_out::[int]
                The number of units in the Linear layer which has weight 
                matrix with shape as in the parameter.
        """
        assert len(shape) == 2, \
               f"{self.__class__.__name__} expected parameter 'shape' to length " + \
               f"of 2. Got shape = {shape} with len(shape) = {len(shape)}."
        
        fan_out, fan_in = shape
        return (fan_in, fan_out)
    
    
    
class XavierUniform(Initializer):
    """Class inheriting from Initializer and overriding _generate method such
    that initialized tensor will follow a Xavier Uniform distribution. 
    Only suitable for weights initialization (not bias initialization).
    Often used when activation function of the layer is Tanh or Sigmoid."""
    
    def _generate(self, shape):
        """Function overriding _generate method of Initializer such that 
        initialized tensor will follow a Xavier Uniform distribution.
        Args:
            shape::[tuple]
                The shape of the tensor we want to create and initialize.
        Returns:
            initialized_tensor::[torch.Tensor]
                The tensor created and initialized via Xavier Uniform.
        """
        fan_in, fan_out = super()._fan_in_fan_out_from_shape(shape)
        bound = math.sqrt( 6 / (fan_out + fan_in) )
        return empty(shape).uniform_(-bound, bound)
    

class XavierNormal(Initializer):
    """Class inheriting from Initializer and overriding _generate method such
    that initialized tensor will follow a Xavier Normal distribution. 
    Only suitable for weights initialization (not bias initialization).
    Often used when activation function of the layer is Tanh or Sigmoid."""
    
    def _generate(self, shape):
        """Function overriding _generate method of Initializer such that 
        initialized tensor will follow a Xavier Normal distribution.
        Args:
            shape::[tuple]
                The shape of the tensor we want to create and initialize.
        Returns:
            initialized_tensor::[torch.Tensor]
                The tensor created and initialized via Xavier Normal.
        """
        fan_in, fan_out = super()._fan_in_fan_out_from_shape(shape)
        std = math.sqrt( 2 / (fan_out + fan_in) )
        return empty(shape).normal_(0., std)


class HeUniform(Initializer):
    """Class inheriting from Initializer and overriding _generate method such
    that initialized tensor will follow a He Uniform distribution. 
    Only suitable for weights initialization (not bias initialization).
    Often used when activation function of the layer is ReLU."""
    
    def _generate(self, shape):
        """Function overriding _generate method of Initializer such that 
        initialized tensor will follow a He Uniform distribution.
        Args:
            shape::[tuple]
                The shape of the tensor we want to create and initialize.
        Returns:
            initialized_tensor::[torch.Tensor]
                The tensor created and initialized via He Uniform.
        """
        fan_in, fan_out = super()._fan_in_fan_out_from_shape(shape)
        bound = math.sqrt( 6 / fan_in )
        return empty(shape).uniform_(-bound, bound)    

    
class HeNormal(Initializer):
    """Class inheriting from Initializer and overriding _generate method such
    that initialized tensor will follow a He Normal distribution. 
    Only suitable for weights initialization (not bias initialization).
    Often used when activation function of the layer is ReLU."""
    
    def _generate(self, shape):
        """Function overriding _generate method of Initializer such that 
        initialized tensor will follow a He Normal distribution.
        Args:
            shape::[tuple]
                The shape of the tensor we want to create and initialize.
        Returns:
            initialized_tensor::[torch.Tensor]
                The tensor created and initialized via He Normal.
        """
        fan_in, fan_out = super()._fan_in_fan_out_from_shape(shape)
        fan_out, fan_in = shape
        return empty(shape).normal_() * math.sqrt(2 / fan_in)
    

class RandomUniform(Initializer):
    """Class inheriting from Initializer and overriding _generate method such
    that initialized tensor will follow a Uniform distribution in the desired
    range."""
    
    def __init__(self, lower = -0.05, upper = +0.05):
        """Constructor storing lower and upper bound of uniform distribution
        used to initialize weight or bias. Makes a sanity check to verify
        that lower bound is smaller or equal to upper bound.
        Args:
            lower::[float]
                The lower bound of the uniform distribution.
            upper::[float]
                The upper bound of the uniform distribution
        """
        assert lower <= upper, \
               f"{self.__class__.__name__} expects parameter 'lower' smaller " + \
               f"or equal than parameter 'upper'. Got: lower = {lower}, upper = {upper}."
               
        self.lower = lower
        self.upper = upper
    
    def _generate(self, shape):
        """Function overriding _generate method of Initializer such that 
        initialized tensor will follow a Uniform distribution in the range
        specified in the constructor.
        Args:
            shape::[tuple]
                The shape of the tensor we want to create and initialize.
        Returns:
            initialized_tensor::[torch.Tensor]
                The tensor created and initialized using a uniform distribution.
        """
        return empty(shape).uniform_(self.lower, self.upper)
    
    
class RandomNormal(Initializer):
    """Class inheriting from Initializer and overriding _generate method such
    that initialized tensor will follow a Normal distribution with desired mean
    and standard deviation."""
    
    def __init__(self, mean = 0., std = 0.05):
        """Constructor storing mean and standard deviation of normal distribution
        used to initialize weight or bias. Makes a sanity check to verify
        that standard deviation is strictly positive.
        Args:
            mean::[float]
                The mean of the normal distribution.
            std::[float]
                The standard deviation of the normal distribution.
        """
        assert std > 0., \
               f"{self.__class__.__name__} expects parameter 'std' strictly " + \
               f"larger than 0. Got: std = {std}."
               
        self.mean = mean
        self.std  = std
        
    def _generate(self, shape):
        """Function overriding _generate method of Initializer such that 
        initialized tensor will follow a Normal distribution with mean and
        standard deviation specified in the constructor.
        Args:
            shape::[tuple]
                The shape of the tensor we want to create and initialize.
        Returns:
            initialized_tensor::[torch.Tensor]
                The tensor created and initialized using a normal distribution.
        """
        return empty(shape).normal_(self.mean, self.std)
        
        
class Zeros(Initializer):
    """Class inheriting from Initializer and overriding _generate method such
    that initialized tensor will be filled with zeros."""
    
    def _generate(self, shape):
        """Function overriding _generate method of Initializer such that 
        initialized tensor will be filled with zeros.
        Args:
            shape::[tuple]
                The shape of the tensor we want to create and initialize.
        Returns:
            initialized_tensor::[torch.Tensor]
                The tensor created and initialized filling it with zeros.
        """
        return empty(shape).fill_(0.)

    
class Ones(Initializer):
    """Class inheriting from Initializer and overriding _generate method such
    that initialized tensor will be filled with ones."""
    
    def _generate(self, shape):
        """Function overriding _generate method of Initializer such that 
        initialized tensor will be filled with ones.
        Args:
            shape::[tuple]
                The shape of the tensor we want to create and initialize.
        Returns:
            initialized_tensor::[torch.Tensor]
                The tensor created and initialized filling it with ones.
        """
        return empty(shape).fill_(1.)

    
class Constant(Initializer):
    """Class inheriting from Initializer and overriding _generate method such
    that initialized tensor will be filled with a desired value."""
    
    def __init__(self, value = 0.):
        """Constructor storing value used to fill initialized weight or bias. 
        Args:
            value::[float]
                The value to fill the initialized tensor with.
        """
        self.value = value
    
    def _generate(self, shape):
        """Function overriding _generate method of Initializer such that 
        initialized tensor will be filled with value specified in the constructor.
        Args:
            shape::[tuple]
                The shape of the tensor we want to create and initialize.
        Returns:
            initialized_tensor::[torch.Tensor]
                The tensor created and initialized filling it with desired value.
        """
        return empty(shape).fill_(self.value)


def get_initializer_instance(initializer):
    """Function returning the parameter if it is already an instance of Initializer
    or if parameter is a string returns instance of the default initializer described
    by the string. Raises an exception if the initializer in the string is unkown
    or if parameter is not of type Initializer nor string.
    Args:
        initializer::[Initializer or string]
            The initializer or a string describing the initializer to be used
            for weight or bias initialization.
    Returns:
        initializer::[Initializer]
            Instance of Initializer to be used for weight or bias initialization.
    """
    if isinstance(initializer, Initializer):
        return initializer
    
    elif isinstance(initializer, str):
        if initializer   == "xavier_uniform":
            return XavierUniform()
        elif initializer == "xavier_normal":
            return XavierNormal()
        elif initializer == "he_uniform":
            return HeUniform()
        elif initializer == "he_normal":
            return HeNormal()
        elif initializer == "random_uniform":
            return RandomUniform()
        elif initializer == "random_normal":
            return RandomNormal()
        elif initializer == "zeros":
            return Zeros()
        elif initializer == "ones":
            return Ones()
        elif initializer == "constant":
            return Constant()
        else:
            raise ValueError("Unknown initializer:", initializer)
    
    else:
        raise TypeError("Initializer parameter type must be str or Initializer. Found:", type(initializer))
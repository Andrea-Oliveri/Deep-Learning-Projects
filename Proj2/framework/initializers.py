# -*- coding: utf-8 -*-

import math
from torch import empty


class Initializer(object):
    
    def __init__(self):
        assert type(self) != Initializer, "Abstract Class Initializer can't be instanciated."
    
    def __call__(self, shape):
        return self._generate(shape)
    
    def _generate(self, shape):
        return
    
    
    
class XavierUniform(Initializer):
    
    def _generate(self, shape):
        assert len(shape) == 2, f"{self.__class__.__name__} expected parameter 'shape' to length of 2. " + \
                                f"Got shape = {shape} with len(shape) = {len(shape)}."
        
        fan_out, fan_in = shape
        bound = math.sqrt( 6 / fan_out + fan_in )
        return empty(shape).uniform_(-bound, bound)
    

class XavierNormal(Initializer):
    
    def _generate(self, shape):
        assert len(shape) == 2, f"{self.__class__.__name__} expected parameter 'shape' to length of 2. " + \
                                f"Got shape = {shape} with len(shape) = {len(shape)}."
        
        fan_out, fan_in = shape
        std = math.sqrt( 2 / fan_out + fan_in )
        return empty(shape).normal_(0., std)


class HeUniform(Initializer):
    
    def _generate(self, shape):
        assert len(shape) == 2, f"{self.__class__.__name__} expected parameter 'shape' to length of 2. " + \
                                f"Got shape = {shape} with len(shape) = {len(shape)}."
        
        fan_out, fan_in = shape
        bound = math.sqrt( 6 / fan_in )
        return empty(shape).uniform_(-bound, bound)    

    
class HeNormal(Initializer):
    
    def _generate(self, shape):
        assert len(shape) == 2, f"{self.__class__.__name__} expected parameter 'shape' to length of 2. " + \
                                f"Got shape = {shape} with len(shape) = {len(shape)}."
        
        fan_out, fan_in = shape
        return empty(shape).normal_() * math.sqrt(2 / fan_in)
    

class RandomUniform(Initializer):
    
    def __init__(self, lower = -0.05, upper = +0.05):
        assert lower <= upper, f"{self.__class__.__name__} expects parameter 'lower' smaller or equal than parameter 'upper'. " + \
                               f"Got: lower = {lower}, upper = {upper}."
        self.lower = lower
        self.upper = upper
    
    def _generate(self, shape):
        return empty(shape).uniform_(self.lower, self.upper)
    
    
class RandomNormal(Initializer):
    
    def __init__(self, mean = 0., std = 0.05):
        assert std > 0., f"{self.__class__.__name__} expects parameter 'std' strictly larger than 0. " + \
                         f"Got: std = {std}."
        self.mean = mean
        self.std  = std
        
    def _generate(self, shape):
        return empty(shape).normal_(self.mean, self.std)
        
        
class Zeros(Initializer):
    
    def _generate(self, shape):
        return empty(shape).fill_(0.)

    
class Ones(Initializer):
    
    def _generate(self, shape):
        return empty(shape).fill_(1.)

    
class Constant(Initializer):
    
    def __init__(self, value = 0.):
        self.value = value
    
    def _generate(self, shape):
        return empty(shape).fill_(self.value)


def get_initializer_instance(initializer):
    if isinstance(initializer, Initializer):
        return initializer
    
    if isinstance(initializer, str):
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
    

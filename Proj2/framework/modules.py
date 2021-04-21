# -*- coding: utf-8 -*-

from torch import empty
from initializers import get_initializer_instance


class Module(object):
    
    def __init__(self):
        raise NotImplementedError
    
    def forward (self, *input):
        raise NotImplementedError
    
    def backward (self, *gradwrtoutput):
        raise NotImplementedError
    
    def param (self):
        return NotImplementedError
    
    
class Linear(Module):
    
    def __init__(units, use_bias = True, activation = None, 
                 kernel_initializer ="glorot_uniform", bias_initializer = "zeros"):
        
        kernel_initializer_instance = get_initializer_instance(kernel_initializer)
        bias_initializer_instance   = get_initializer_instance(bias_initializer)
        
    pass


class ReLU(Module):
    pass


class Tanh(Module):
    pass
# -*- coding: utf-8 -*-

from torch import empty
from modules import Module


class Sequential(Module):
    
    def __init__(self): # learning rate, optimizer ?
        self.layers = []
        
    def __init__(self, *layers):
        assert all([issubclass(layer, Module) for layer in layers]), "The layers should be subclass of Module."
        self.layers = list(layers)
    
    def add(self, module):
        assert issubclass(module, Module), "The layers should be subclass of Module."
        self.layers.append(module)
    
    def forward(self, *inputs):
        outputs = inputs
        
        for layer in self.layers:
            outputs = layer.forward(*outputs)
            
        return outputs
    
    def backward(self, *gradwrtoutput):
        gradwrtinput = gradwrtoutput
        
        for layer in self.layers[::-1]:
            gradwrtinput = layer.backward(*gradwrtinput)
            
    def update_params(self, lr):
        for layer in self.layers:
            layer.update_params(lr)
        
    def param(self):
        return [layer.params() for layer in self.layers]
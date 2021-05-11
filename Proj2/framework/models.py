# -*- coding: utf-8 -*-

from torch import empty
from .modules import Module


class Sequential(Module):
    
    def __init__(self): # learning rate, optimizer ?
        self.layers = []
        
    def __init__(self, *layers):
        # If the layers were given as a single list or tuple rather than as args.
        if len(layers) == 1 and type(layers[0]) in (tuple, list):
            layers = layers[0]
            
        assert all([isinstance(layer, Module) for layer in layers]), "The layers should be an instance of a subclass of Module."
        self.layers = list(layers)
    
    def add(self, layer):
        assert isinstance(layer, Module), "The layer should be an instance of a subclass of Module."
        self.layers.append(layer)
    
    def forward(self, *inputs):
        outputs = inputs
        
        for layer in self.layers:
            outputs = layer.forward(*outputs)
            
        return tuple(outputs)
    
    def backward(self, *gradwrtoutput):
        gradwrtinput = gradwrtoutput
        
        for layer in self.layers[::-1]:
            gradwrtinput = layer.backward(*gradwrtinput)
            
            
        # THIS IS JUST FOR DEBUG. REMOVE AT END
        return gradwrtinput
    
            
    def update_params(self, lr):
        for layer in self.layers:
            layer.update_params(lr)
        
    def param(self):
        return [layer.param() for layer in self.layers]
    
    def load_params(self, params):
        assert len(params) == len(self.layers),  "Parameters to load in Sequential model should be a " + \
                                                f"list of length len(n_layers) = {len(self.layers)}. " + \
                                                f"Got len(parameters_to_load) = {len(params)}."
        for layer, layer_params in zip(self.layers, params):
            layer.load_params(layer_params)
# -*- coding: utf-8 -*-

from .module import Module


class Sequential(Module):
    
    def __init__(self, *layers):
        """Initialize the model from an arbitratry number of layers given as
        parameters.
        Args:
            layers::[tuple]
                Arbitrary number of layers to add in the model in order. It 
                should be noted that we can only have subclasses of Module in 
                this tuple as Module is not considered as a layer.
        """
        # If the layers were given as a single list or tuple rather than as args.
        if len(layers) == 1 and type(layers[0]) in (tuple, list):
            layers = layers[0]
            
        assert all([isinstance(layer, Module) for layer in layers]), \
               "The layers should be an instance of a subclass of Module."
              
        self.layers = list(layers)
    
    def add(self, layer):
        """Add a layer at the end of the model.
        Args:
            layer::[subclass(Module)]
                The layer to add at the end of the model. It should be noted
                that we can only add subclasses of Module as Module is not
                considered as a layer.
        """
        assert isinstance(layer, Module), \
               "The layer should be an instance of a subclass of Module."
               
        self.layers.append(layer)
    
    def forward(self, *inputs):
        """Perform the forward pass for each input: compute the outputs of the 
        model given inputs.
        Args:
            inputs::[tuple]
                Tuple containing input tensors on which we wish to perform the 
                forward pass and from which we want to compute the outputs.
        Returns:
            outputs::[tuple]
                Tuple containing the result of the forward method applied on
                inputs parameter.
        """
        outputs = inputs
        
        for layer in self.layers:
            outputs = layer.forward(*outputs)
            
        return tuple(outputs)
    
    def backward(self, *gradwrtoutput):
        """TO DOOOOOOOOOOOOOOOOOOOOOOOOOOOO
        Args:
            gradwrtoutput::[tuple]
                Tuple containing input tensors on which we wish to perform the 
                backward pass and from which we want to compute the outputs.
        Returns:
            gradwrtinput::[tuple]
                Tuple containing the result of the backward method applied on
                inputs parameter.
        """
        gradwrtinput = gradwrtoutput
        
        for layer in self.layers[::-1]:
            gradwrtinput = layer.backward(*gradwrtinput)
        
    def param(self):
        """Return the list of the parameters of the model as well as their
        corresponding gradient.
        Returns:
            params::[list]
                List of pairs, each composed of a parameter tensor, and its 
                corresponding gradient tensor of same size.
        """
        return [layer.param() for layer in self.layers]
    
    def update_params(self, lr):
        """Update the parameters of the model given their gradients stored as 
        attribute and the learning rate given as parameter. 
        Args:
            lr::[float]
                The learning rate by which we want to update the parameters.
        """
        for layer in self.layers:
            layer.update_params(lr)
    
    def load_params(self, params):
        assert len(params) == len(self.layers), \
                "Parameters to load in Sequential model should be a " + \
               f"list of length len(n_layers) = {len(self.layers)}. " + \
               f"Got len(parameters_to_load) = {len(params)}."
               
        for layer, layer_params in zip(self.layers, params):
            layer.load_params(layer_params)
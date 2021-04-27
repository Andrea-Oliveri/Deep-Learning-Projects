# -*- coding: utf-8 -*-

from torch import empty
from initializers import get_initializer_instance


class Module(object):
    
    def __init__(self):
        assert type(self) != Module, "Abstract Class Module can't be instanciated."
    
    def forward(self, *inputs):
        raise NotImplementedError
    
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    
    def param(self):
        return []
    
    
class Linear(Module):
    
    def __init__(fan_in, fan_out, use_bias = True, 
                 weight_initializer = "xavier_uniform", bias_initializer = "zeros"):
        
        weight_initializer_instance = get_initializer_instance(weight_initializer)
        self.weight = weight_initializer_instance((fan_out, fan_in))
        self.grad_weight = empty(self.weight.shape).zero_()
        
        self.use_bias = use_bias
        
        if use_bias:
            bias_initializer_instance = get_initializer_instance(bias_initializer)
            self.bias = bias_initializer_instance((fan_out, 1))
            self.grad_bias = empty(self.bias.shape).zero_()

        
        
    def forward(self, *inputs):
        self.forward_pass_inputs = inputs
        
        outputs = []
        for input_tensor in inputs:
            output = self.weight @ input_tensor
            
            if self.use_bias:
                output += self.bias
                
            outputs.append( output )
            
        return tuple(outputs)
    
    
    def backward(self, *gradwrtoutput):
        gradwrtinput = []
        for forward_pass_input, grad_output in zip(self.forward_pass_inputs, gradwrtoutput):
            self.grad_weight += forward_pass_input.T @ grad_output
            
            if self.use_bias:
                self.grad_bias += grad_output

            gradwrtinput.append( self.weights.T @ grad_output )
                
        return tuple(gradwrtinput)
    
    
    def param(self):
        parameters = [(self.weight, self.grad_weight)]
        
        if self.use_bias:
            parameters.append( (self.bias, self.grad_bias) )
            
        return parameters
    
    
    def sgd_step(self, lr):
        self.weight -= lr * self.grad_weight
        
        if self.use_bias:
            self.bias -= lr * self.grad_bias
        
    


class ReLU(Module):
        
    def forward(self, *inputs):
        outputs = []
        for input_tensor in inputs:
            outputs.append( input_tensor.relu() )
            
        return tuple(outputs)
    
    def backward(self, *gradwrtoutput):
        raise NotImplementedError


class Tanh(Module):
        
    def forward(self, *inputs):
        outputs = []
        for input_tensor in inputs:
            outputs.append( input_tensor.tanh() )
            
        return tuple(outputs)
    
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
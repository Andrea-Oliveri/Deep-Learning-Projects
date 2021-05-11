# -*- coding: utf-8 -*-

from torch import empty
from .initializers import get_initializer_instance


class Module(object):
    
    def __init__(self):
        assert type(self) != Module, "Abstract Class Module can't be instanciated."
        
    def __call__(self, *inputs):
        return self.forward(*inputs)
    
    def forward(self, *inputs):
        raise NotImplementedError
    
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    
    def param(self):
        return []
    
    def update_params(self, lr):
        return
    
    def load_params(self, params):
        return
    
    
class Linear(Module):
    
    def __init__(self, fan_in, fan_out, use_bias = True, 
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
            assert input_tensor.ndim == 2 and input_tensor.shape[-1] == 1, f"Linear layer expects input of shape (n_dims, 1). Got {input_tensor.shape}"

            output = self.weight @ input_tensor
            
            if self.use_bias:
                output += self.bias
                
            outputs.append( output )
            
        return tuple(outputs)
    
    def backward(self, *gradwrtoutput):
        gradwrtinput = []
        for forward_pass_input, grad_output in zip(self.forward_pass_inputs, gradwrtoutput):
            self.grad_weight += grad_output @ forward_pass_input.T 
            
            if self.use_bias:
                self.grad_bias += grad_output

            gradwrtinput.append( self.weight.T @ grad_output )
                
        return tuple(gradwrtinput)
    
    def param(self):
        parameters = [(self.weight, self.grad_weight)]
        
        if self.use_bias:
            parameters.append( (self.bias, self.grad_bias) )
            
        return parameters
    
    def update_params(self, lr):
        self.weight -= lr * self.grad_weight
        self.grad_weight = empty(self.weight.shape).zero_()
        
        if self.use_bias:
            self.bias -= lr * self.grad_bias
            self.grad_bias = empty(self.bias.shape).zero_()
            
    def load_params(self, params):
        if self.use_bias:
            assert len(params) == 2, "Linear layer using bias expects params to be loaded to be " + \
                                     "a list containing two elements: [weight, bias]. " + \
                                     "Got len(params) = {}".format(len(params))
                                     
            (new_weight, new_grad_weight), (new_bias, new_grad_bias) = params
                                      
            assert new_bias.shape == self.bias.shape,  "Shape mismatch between size of bias to be " + \
                                                       "loaded by Linear layer and expected size. " + \
                                                      f"Got new_bias.shape = {new_bias.shape} " + \
                                                      f"and expected_shape = {self.bias.shape}."
                                                      
            assert new_grad_bias.shape == self.grad_bias.shape, \
                    "Shape mismatch between size of gradient bias to be loaded by Linear layer " + \
                   f"and expected size. Got new_grad_bias.shape = {new_grad_bias.shape} and " + \
                   f"expected_shape = {self.grad_bias.shape}."
            
            self.bias      = new_bias
            self.grad_bias = new_grad_bias
            
        else:
            assert len(params) == 1, "Linear layer not using bias expects params to be loaded to be " + \
                                     "a list containing one elements: [weight]. " + \
                                     "Got len(params) = {}".format(len(params))
        
            (new_weight, new_grad_weight), = params
        
        assert new_weight.shape == self.weight.shape,  "Shape mismatch between size of weight to be " + \
                                                       "loaded by Linear layer and expected size. " + \
                                                      f"Got new_weight.shape = {new_weight.shape} " + \
                                                      f"and expected_shape = {self.weight.shape}."
        
        assert new_grad_weight.shape == self.grad_weight.shape, \
                    "Shape mismatch between size of gradient weight to be loaded by Linear layer " + \
                   f"and expected size. Got new_grad_weight.shape = {new_grad_weight.shape} and " + \
                   f"expected_shape = {self.grad_weight.shape}."
            
        self.weight      = new_weight
        self.grad_weight = new_grad_weight
                    


class ReLU(Module):
        
    def forward(self, *inputs):
        self.forward_pass_inputs = inputs
        
        outputs = []
        for input_tensor in inputs:
            outputs.append( input_tensor.relu() )
            
        return tuple(outputs)
    
    def backward(self, *gradwrtoutput):
        gradwrtinput = []
        
        for forward_pass_input, grad_output in zip(self.forward_pass_inputs, gradwrtoutput):
            gradwrtinput.append( (forward_pass_input > 0) * grad_output )
                
        return tuple(gradwrtinput)


class Tanh(Module):
        
    def forward(self, *inputs):
        self.forward_pass_inputs = inputs

        outputs = []
        for input_tensor in inputs:
            outputs.append( input_tensor.tanh() )
            
        return tuple(outputs)
    
    def backward(self, *gradwrtoutput):
        gradwrtinput = []
        
        for forward_pass_input, grad_output in zip(self.forward_pass_inputs, gradwrtoutput):
            gradwrtinput.append( (1 / forward_pass_input.cosh().pow(2)) * grad_output )
                
        return tuple(gradwrtinput)
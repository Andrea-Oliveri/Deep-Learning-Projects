# -*- coding: utf-8 -*-

from torch import empty
from .initializers import get_initializer_instance


class Module(object):
    """Abstract Class from which different modules included the different 
    models, layers and losses will inherit from."""
    
    def __init__(self):
        """Stub constructor of Module. Raises an AssertionError as the 
        abstract class Module can't be instanciated."""
        assert type(self) != Module, \
               "Abstract Class Module can't be instanciated."
        
    def __call__(self, *inputs):
        """Special method defined for conveniently calling the forward method.
        Args:
            inputs::[tuple]
                See forward for the explaination of the inputs parameter.
        Returns:
            outputs::[torch.Tensor]
                See forward for the explaination of the returned value outputs.
        """
        return self.forward(*inputs)
    
    def forward(self, *inputs):
        """Stub function. Child classes will override it such that it will
        perform the forward pass for each input.
        Args:
            inputs::[tuple]
                Tuple containing an arbitrary number of input tensor on which
                we wish to perform the forward pass.
        Returns:
            outputs::[tuple]
                Tuple containing the result of the forward method applied on
                inputs parameter. The treatment depends on the child class.
        """
        return
    
    def backward(self, *gradwrtoutput):
        """Stub function. Child classes will override it such that it will
        perform the backward pass for each gradient with respect to the outputs
        of the current module which were computed during the forward pass.
        Args:
            gradwrtoutput::[tuple]
                Tuple containing an arbitrary number of tensors on which we
                wish to perform the backward pass.
        Returns:
            gradwrtinput::[tuple]
                Tuple containing the result of the backward method applied on
                inputs parameter. The treatment depends on the child class.
        """
        return
    
    def param(self):
        """Some child classes will override this method such that it will 
        return the list of the parameters of the modelas well as their
        corresponding gradient. If there is no parameter for some subclass, it 
        will just use this one and return an empty list.
        Returns:
            params::[list]
                List of pairs, each composed of a parameter tensor, and its 
                corresponding gradient tensor of same size.
        """
        return []
    
    def update_params(self, lr):
        """Some child classes will override this method such that it will 
        update its parameters given its gradient stored as attribute and the 
        learning rate given as parameter. If there is no parameter for some 
        subclass, no update of parameters is needed and then it will just use 
        this method and do nothing.
        Args:
            lr::[float]
                The learning rate by which we want to update the parameters.
        """
        return
    
    def load_params(self, params):
        """Some child classes will override this method such that it will 
        replace its parameters with the parameters given as parameter. It must
        be noted that the params parameter should have the same structure as
        the structure of the parameters of the model. If there is no parameter 
        for some subclass, no loading of parameters is needed and then it will 
        just use this method and do nothing.
        Args:
            params::[list]
                List of pairs, each composed of a parameter tensor, and its 
                corresponding gradient tensor of same size.
        """
        return
    
    
class Linear(Module):
    """Class inheriting from Module and overriding several methods so that it
    represents a linear layer of a model"""
    
    def __init__(self, fan_in, fan_out, use_bias = True, 
                 weight_initializer = "xavier_uniform", bias_initializer = "zeros"):
        """This method will initialize the Linear layer given the number of 
        units in the previous and in the current layer, the initializer to use 
        in order to initialize its parameters and the use of a bias.
        Args:
            fan_in::[int]
                The number of input units from the previous layer.
            fan_out::[int]
                The number of units in this layer.
            use_bias::[bool]
                Boolean that defines if this Linear layer will also have a bias 
                as parameter.
            weight_initializer::[str]
                The name of the initializer to use to initialize the weight
                parameter.
            bias_initializer::[str]
                If a bias is used, the name of the initializer to use to
                initialize the bias parameter.
        """
        weight_initializer_instance = get_initializer_instance(weight_initializer)
        self.weight = weight_initializer_instance((fan_out, fan_in))
        self.grad_weight = empty(self.weight.shape).zero_()
        
        self.use_bias = use_bias
        
        if use_bias:
            bias_initializer_instance = get_initializer_instance(bias_initializer)
            self.bias = bias_initializer_instance((fan_out, 1))
            self.grad_bias = empty(self.bias.shape).zero_()

    def forward(self, *inputs):
        """Perform the forward pass for each input: compute the outputs of the 
        layer given inputs.
        Args:
            inputs::[tuple]
                Tuple containing input tensors on which we wish to perform the 
                forward pass and from which we want to compute the outputs.
        Returns:
            outputs::[tuple]
                Tuple containing the result of the forward method applied on
                inputs parameter.
        """
        self.forward_pass_inputs = inputs
        
        outputs = []
        for input_tensor in inputs:
            assert input_tensor.ndim == 2 and input_tensor.shape[-1] == 1, \
                   "Linear layer expects input of shape (n_dims, 1). " + \
                   f"Got {input_tensor.shape}"

            output = self.weight @ input_tensor
            
            if self.use_bias:
                output += self.bias
                
            outputs.append( output )
            
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
        gradwrtinput = []
        for forward_pass_input, grad_output in zip(self.forward_pass_inputs, gradwrtoutput):
            self.grad_weight += grad_output @ forward_pass_input.T 
            
            if self.use_bias:
                self.grad_bias += grad_output

            gradwrtinput.append( self.weight.T @ grad_output )
                
        return tuple(gradwrtinput)
    
    def param(self):
        """Return the list of the parameters of the layer as well as their
        corresponding gradient.
        Returns:
            params::[list]
                List of pairs, each composed of a parameter tensor, and its 
                corresponding gradient tensor of same size.
        """
        parameters = [(self.weight, self.grad_weight)]
        
        if self.use_bias:
            parameters.append( (self.bias, self.grad_bias) )
            
        return parameters
    
    def update_params(self, lr):
        """Update the parameters of the layer given their gradient stored as 
        attribute and the learning rate given as parameter. 
        Args:
            lr::[float]
                The learning rate by which we want to update the parameters.
        """
        self.weight -= lr * self.grad_weight
        self.grad_weight = empty(self.weight.shape).zero_()
        
        if self.use_bias:
            self.bias -= lr * self.grad_bias
            self.grad_bias = empty(self.bias.shape).zero_()
            
    def load_params(self, params):
        """Replace the parameters of the layer by the parameters given as 
        parameter. It must be noted that the params parameter should have the 
        same structure as the structure of the parameters of this layer,
        otherwise it will raise an error.
        Args:
            params::[list]
                List of pairs, each composed of a parameter tensor, and its 
                corresponding gradient tensor of same size.
        """
        if self.use_bias:
            assert len(params) == 2, \
                   "Linear layer using bias expects params to be loaded to be " + \
                   "a list containing two elements: [weight, bias]. " + \
                   "Got len(params) = {}".format(len(params))
                                     
            (new_weight, new_grad_weight), (new_bias, new_grad_bias) = params
                                      
            assert new_bias.shape == self.bias.shape, \
                    "Shape mismatch between size of bias to be " + \
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
            assert len(params) == 1, \
                   "Linear layer not using bias expects params to be loaded to be " + \
                   "a list containing one elements: [weight]. " + \
                   "Got len(params) = {}".format(len(params))
        
            (new_weight, new_grad_weight), = params
        
        assert new_weight.shape == self.weight.shape, \
                "Shape mismatch between size of weight to be " + \
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
        """Perform the forward pass for each input: compute the outputs of the 
        layer given inputs. Here it just corresponds to apply the ReLU
        activation function on the inputs.
        Args:
            inputs::[tuple]
                Tuple containing input tensors on which we wish to perform the 
                forward pass and from which we want to compute the outputs.
        Returns:
            outputs::[tuple]
                Tuple containing the result of the forward method applied on
                inputs parameter.
        """
        self.forward_pass_inputs = inputs
        
        outputs = []
        for input_tensor in inputs:
            outputs.append( input_tensor.relu() )
            
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
        gradwrtinput = []
        
        for forward_pass_input, grad_output in zip(self.forward_pass_inputs, gradwrtoutput):
            gradwrtinput.append( (forward_pass_input > 0) * grad_output )
                
        return tuple(gradwrtinput)


class Tanh(Module):
        
    def forward(self, *inputs):
        """Perform the forward pass for each input: compute the outputs of the 
        layer given inputs. Here it just corresponds to apply the Tanh
        activation function on the inputs.
        Args:
            inputs::[tuple]
                Tuple containing input tensors on which we wish to perform the 
                forward pass and from which we want to compute the outputs.
        Returns:
            outputs::[tuple]
                Tuple containing the result of the forward method applied on
                inputs parameter.
        """
        self.forward_pass_inputs = inputs

        outputs = []
        for input_tensor in inputs:
            outputs.append( input_tensor.tanh() )
            
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
        gradwrtinput = []
        
        for forward_pass_input, grad_output in zip(self.forward_pass_inputs, gradwrtoutput):
            gradwrtinput.append( (1 / forward_pass_input.cosh().pow(2)) * grad_output )
                
        return tuple(gradwrtinput)
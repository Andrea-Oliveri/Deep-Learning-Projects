# -*- coding: utf-8 -*-

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
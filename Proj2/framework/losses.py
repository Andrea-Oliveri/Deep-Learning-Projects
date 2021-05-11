# -*- coding: utf-8 -*-

from .modules import Module

class Loss(Module):
    
    def __init__(self):
        assert type(self) != Loss, "Abstract Class Loss can't be instanciated."
        
        
    def forward(self, target, prediction):
        return
     
        
    def backward(self, target, prediction):
        return
    


class LossMSE(Loss):
    
    def forward(self, target, prediction):
        assert target.shape == prediction.shape,  "MSE requires target and prediction tensors to have same shape. " + \
                                                 f"Got target.shape = {target.shape}, prediction.shape = {prediction.shape}"

        return (target - prediction).pow(2).mean()
        
        
    def backward(self, target, prediction):
        """ Returns only one gradient even though forward takes two: only gradient wrt predictions. SGD!!
        Args:
            -
            
        Returns:
            - 
        """
        return (target - prediction).multiply_(-2 / len(target))
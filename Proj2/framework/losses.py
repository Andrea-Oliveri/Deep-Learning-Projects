# -*- coding: utf-8 -*-

from modules import Module
    
    
class MSE(Module):
    
    def forward(self, *inputs):
    """First param in inputs is target, second param is prediction"""
        assert len(inputs) == 2, f"MSE only takes two parameters: (target_tensor, prediction_tensor). Got {len(inputs)} inputs."
        target, pred = inputs
        assert target.shape == pred.shape, f"MSE requires target and prediction tensors to have same shape. " + \
                                           f"Got target.shape = {target.shape}, prediction.shape = {pred.shape}"

        return ((target - pred)**2).mean()
        
        
    def backward(self, *gradwrtoutput):
    """Returns only one gradient even though forward takes two: only gradient wrt predictions. SGD!!"""    
        pass
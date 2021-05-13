# -*- coding: utf-8 -*-

from .module import Module


class Loss(Module):
    """Abstract Class inheriting from Module Class and from which the different 
    the losses will inherit from."""
    
    def __init__(self):
        """Stub constructor of Loss. Raises an AssertionError as the 
        abstract class Loss can't be instanciated."""
        assert type(self) != Loss, \
               "Abstract Class Loss can't be instanciated."
        
        
    def forward(self, target, prediction):
        """Stub function. Child classes will override it such that it will
        compute the loss with a different formula for each child class.
        Args:
            target::[torch.Tensor]
                The target tensor in one hot format.
            prediction::[torch.Tensor]
                The predicted tensor in one hot format, its shape must be the
                same as the shape of the target.
        Returns:
            computed_loss::[torch.Tensor]
                0d tensor containig the computed loss.
        """
        return
     
        
    def backward(self, target, prediction):
        """Stub function. Child classes will override it such that it will
        compute the gradient of the loss with respect to prediction with a 
        different formula for each child class.
        Args:
            target::[torch.Tensor]
                The target tensor in one hot format.
            prediction::[torch.Tensor]
                The predicted tensor in one hot format, its shape must be the
                same as the shape of the target.
        Returns:
            computed_loss_gradient::[torch.Tensor]
                Tensor containig the computed gradient of the loss with 
                respect to the prediction. It will have the same shape as the 
                target and prediction.
        """
        return
    


class LossMSE(Loss):
    """Class inheriting from Loss and overriding forward and backward methods 
    such that it will take the MSE (Mean Squarred Error) as loss function."""
    
    def forward(self, target, prediction):
        """Compute the MSE loss from the target and prediction tensors. Raises 
        an error if the target and the prediction tensors do not have the same 
        shape.
        Args:
            target::[torch.Tensor]
                The target tensor in one hot format.
            prediction::[torch.Tensor]
                The predicted tensor in one hot format, its shape must be the
                same as the shape of the target.
        Returns:
            computed_loss::[torch.Tensor]
                0d tensor containig the computed MSE loss.
        """
        assert target.shape == prediction.shape, \
                "MSE requires target and prediction tensors to have same shape. " + \
               f"Got target.shape = {target.shape}, prediction.shape = " + \
               f"{prediction.shape}"

        return (target - prediction).pow(2).mean()
        
        
    def backward(self, target, prediction):
        """Compute the MSE gradient with respect to the prediction tensor. 
        Raises an error if the target and the prediction tensors do not have the
        same shape.
        Args:
            target::[torch.Tensor]
                The target tensor in one hot format.
            prediction::[torch.Tensor]
                The predicted tensor in one hot format, its shape must be the
                same as the shape of the target.
        Returns:
            computed_loss_gradient::[torch.Tensor]
                Tensor containig the computed MSE gradient with respect to the 
                prediction. It will have the same shape as the target and 
                prediction.
        """
        assert target.shape == prediction.shape, \
                "Computing the MSE gardient with respect to prediction requires " + \
                "target and prediction tensors to have same shape. " + \
               f"Got target.shape = {target.shape}, prediction.shape = " + \
               f"{prediction.shape}"
               
        return (target - prediction).multiply_(-2 / len(target))
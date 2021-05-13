# -*- coding: utf-8 -*-

import math


class LearningRateScheduler(object):
    """Abstract Class from which different learning rate schedulers will 
    inherit from."""
    
    def __init__(self):
        """Stub constructor of LearningRateScheduler. Raises an AssertionError 
        as the abstract class Initializer can't be instanciated."""
        assert type(self) != LearningRateScheduler, \
               "Abstract Class LearningRateScheduler can't be instanciated."
        
    def __call__(self):
        """Special method defined for conveniently calling the get_new_lr method.
        Returns:
            lr::[float]
                See get_new_lr for the explaination of the returned value lr.
        """
        return self.get_new_lr()
    
    def get_new_lr(self):
        """Stub function. Child classes will override it such that it will
        generate a new learning rate from the initial one stored as attribute 
        with different techniques depending on the child class.
        Returns:
            lr::[float]
                The newly generated learning rate.
        """
        return



class ConstantLR(LearningRateScheduler):
    """Class inheriting from LearningRateScheduler and overriding get_new_lr 
    such that it will generate a new learning rate from a given initial one.
    For this subclass, it is a constant learning rate and then the newly 
    generated learning rate is always the initial one."""
    
    def __init__(self, lr):
        """The constructor will initialize the learning rate of ConstantLr.
        Args:
            lr::[float]
                The initial learning rate of the learning rate scheduler.
        """
        self.lr = lr
        
    def get_new_lr(self):
        """Generate a new learning rate from the initial one stored as 
        attribute. For this subclass, it is always constant and then the returned
        learning rate is always the initial one.
        Returns:
            lr::[float]
                The newly generated learning rate.
        """
        return self.lr
    


class TimeDecayLR(LearningRateScheduler):
    """Class inheriting from LearningRateScheduler and overriding get_new_lr 
    such that it will generate a new learning rate from a given initial one and 
    from a given decay. For this subclass, the newly generated learning rate is
    the initial one with a decay of decay at each iteration."""
    
    def __init__(self, initial_lr, decay):
        """The constructor will initialize the learning rate and the wanted
        decay at each step for TimeDecayLR.
        Args:
            lr::[float]
                The initial learning rate of the learning rate scheduler.
            decay::[float]
                The wanted decay at each step.
        """
        self.initial_lr = initial_lr
        self.decay      = decay
        self.iterations = 0

    def get_new_lr(self):
        """Generate a new learning rate from the initial one, from the decay 
        and from the current iteration all stored as attributes. For this 
        subclass, the learning rate decay of decay at each iteration.
        Returns:
            lr::[float]
                The newly generated learning rate.
        """
        lr = self.initial_lr / (1. + self.decay * self.iterations)
        self.iterations += 1
        return lr
    


class ExponentialDecayLR(LearningRateScheduler):
    """Class inheriting from LearningRateScheduler and overriding get_new_lr 
    such that it will generate a new learning rate from a given initial one and 
    from a given decay. For this subclass, the newly generated learning rate is
    the initial one with an exponential decay of decay at each iteration."""
    
    def __init__(self, initial_lr, decay):
        """The constructor will initialize the learning rate and the wanted
        exponential decay at each step for ExponentialDecayLR.
        Args:
            lr::[float]
                The initial learning rate of the learning rate scheduler.
            decay::[float]
                The wanted exponential decay at each step.
        """
        self.initial_lr = initial_lr
        self.decay      = decay
        self.iterations = 0

    def get_new_lr(self):
        """Generate a new learning rate from the initial one, from the decay 
        and from the current iteration all stored as attributes. For this 
        subclass, the learning rate decay exponentially of decay at each 
        iteration.
        Returns:
            lr::[float]
                The newly generated learning rate.
        """
        lr = self.initial_lr * math.exp(-self.decay * self.iterations)
        self.iterations += 1
        return lr
    
    
    
class StepDecayLR(LearningRateScheduler):
    
    def __init__(self, initial_lr, drop_factor, iterations_drop_period):
        """The constructor will initialize the learning rate and the TO DOOOOOOOOOOOOOOOOOOOOOOO
        at each step for StepDecayLR.
        Args:
            lr::[float]
                The initial learning rate of the learning rate scheduler.
        """
        self.initial_lr  = initial_lr
        self.drop_factor = drop_factor
        self.iterations_drop_period = iterations_drop_period
        self.iterations  = 0

    def get_new_lr(self):
        """Generate a new learning rate from the initial one, from TO DOOOOOOOOOOOOOOOOOOOOOOOO
        Returns:
            lr::[float]
                The newly generated learning rate.
        """
        n_times_dropped = self.iterations // self.iterations_drop_period
        lr = self.initial_lr * (self.drop_factor**n_times_dropped)
        self.iterations += 1
        return lr
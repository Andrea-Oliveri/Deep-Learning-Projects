# -*- coding: utf-8 -*-

import math


class LearningRateScheduler(object):
    def __init__(self):
        assert type(self) != LearningRateScheduler, "Abstract Class LearningRateScheduler can't be instanciated."
        
    def get_new_lr(self):
        return
        
    def __call__(self):
        return self.get_new_lr()



class ConstantLR(LearningRateScheduler):
    
    def __init__(self, lr):
        self.lr = lr
        
    def get_new_lr(self):
        return self.lr
    


class TimeDecayLR(LearningRateScheduler):
    def __init__(self, initial_lr, decay):
        self.initial_lr = initial_lr
        self.decay      = decay
        self.iterations = 0

    def get_new_lr(self):
        lr = self.initial_lr / (1. + self.decay * self.iterations)
        self.iterations += 1
        return lr
    


class ExponentialDecayLR(LearningRateScheduler):
    def __init__(self, initial_lr, decay):
        self.initial_lr = initial_lr
        self.decay      = decay
        self.iterations = 0

    def get_new_lr(self):
        lr = self.initial_lr * math.exp(-self.decay * self.iterations)
        self.iterations += 1
        return lr
    
    
    
class StepDecayLR(LearningRateScheduler):
    def __init__(self, initial_lr, drop_factor, iterations_drop_period):
        self.initial_lr  = initial_lr
        self.drop_factor = drop_factor
        self.iterations_drop_period = iterations_drop_period
        self.iterations  = 0

    def get_new_lr(self):
        n_times_dropped = self.iterations // self.iterations_drop_period
        lr = self.initial_lr * (self.drop_factor**n_times_dropped)
        self.iterations += 1
        return lr
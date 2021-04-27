# -*- coding: utf-8 -*-

from torch import empty
from modules import Module


class Sequential(Module):
    
    def __init__(self):
        pass
    
    def add(self, module):
        pass
    
    def forward(self, *inputs):
        raise NotImplementedError
    
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    
    def param(self):
        return []
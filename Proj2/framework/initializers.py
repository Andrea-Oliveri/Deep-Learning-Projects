# -*- coding: utf-8 -*-

from torch import empty


class Initializer(object):
    
    def __init__(self):
        raise NotImplementedError
    
    def generate(self, shape):
        raise NotImplementedError
    
    def __call__(self, shape):
        return self.generate(shape)
    
    
class GlorotUniform(Initializer):
    pass


class GlorotNormal(Initializer):
    pass


class RandomUniform(Initializer):
    pass


class RandomNormal(Initializer):
    pass


class Zeros(Initializer):
    pass


class Ones(Initializer):
    pass


class Constant(Initializer):
    pass



def get_initializer_instance(initializer):
    if isinstance(initializer, Initializer):
        return initializer
    
    if isinstance(initializer, str):
        if initializer == "glorot_uniform":
            pass
        elif initializer == "glorot_normal":
            pass
        elif initializer == "random_uniform":
            pass
        elif initializer == "random_normal":
            pass
        elif initializer == "zeros":
            pass
        elif initializer == "ones":
            pass
        elif initializer == "constant":
            pass
        else:
            raise ValueError("Unknown initializer:", initializer)
    
    else:
        raise TypeError("Initializer parameter type must be str or Initializer. Found:", type(initializer))
    

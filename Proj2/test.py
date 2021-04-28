# -*- coding: utf-8 -*-

import math
from torch import empty, set_grad_enabled
set_grad_enabled(False)

import framework



def generate_data(shape_training_set = (1000, 2), shape_test_set = (1000, 2)):
    train_input = empty(shape_training_set).uniform_()
    test_input  = empty(shape_test_set    ).uniform_()
        
    train_target = ((train_input - 0.5).pow(2).sum(axis = 1).sqrt() < 1 / math.sqrt(2*math.pi)).int()
    test_target  = ((test_input  - 0.5).pow(2).sum(axis = 1).sqrt() < 1 / math.sqrt(2*math.pi)).int()

    return train_input, train_target, test_input, test_target


train_input, train_target, test_input, test_target = generate_data()


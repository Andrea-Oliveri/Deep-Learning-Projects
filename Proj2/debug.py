# -*- coding: utf-8 -*-

import math
import torch
from torch import empty, tensor, nn

from framework.models import Sequential
from framework.modules import Linear, Relu, Tanh
from framework.losses import MSE

from framework.initializers import XavierUniform, XavierNormal, HeUniform, HeNormal

import matplotlib.pyplot as plt



def generate_data(n_training_samples = 10, n_test_samples = 10):
    all_inputs = empty(n_training_samples + n_test_samples, 2).uniform_()

    all_classes = ((all_inputs - 0.5).pow(2).sum(axis = 1).sqrt() < 1 / math.sqrt(2*math.pi)).long()
    all_target  = empty(n_training_samples + n_test_samples, 2).zero_()
    all_target[all_classes == 0, 0] = 1
    all_target[all_classes == 1, 1] = 1
    
    train_input , test_input  = all_inputs.split([n_training_samples, n_test_samples])
    train_target, test_target = all_target.split([n_training_samples, n_test_samples])

    return train_input, train_target, test_input, test_target


n_tries = 100
train_input, train_target, _, _ = generate_data(n_tries, n_tries)

for input_, target in zip(train_input, train_target):

    input_ = input_.view(-1, 1)
    target = target.view(-1, 1)
    input_.requires_grad = True

    lin_layer1 = Linear(2, 4)
    lin_layer_torch1 = nn.Linear(2, 4)
    
    lin_layer1.weight = lin_layer_torch1.weight.clone().detach()
    lin_layer1.bias   = lin_layer_torch1.bias.  clone().detach().reshape(-1, 1)
    
    lin_layer2 = Linear(4, 2)
    lin_layer_torch2 = nn.Linear(4, 2)
    
    lin_layer2.weight = lin_layer_torch2.weight.clone().detach()
    lin_layer2.bias   = lin_layer_torch2.bias.  clone().detach().reshape(-1, 1)
    
    seq = Sequential(lin_layer1, Relu(), lin_layer2, Tanh())
    seq_torch = nn.Sequential(lin_layer_torch1, nn.ReLU(), lin_layer_torch2, nn.Tanh())

    output, = seq(input_)
    output_torch = seq_torch(input_.T).T
    
    assert torch.allclose(output, output_torch), "Test Same Output FAILED"
    

    
    loss = MSE()
    loss_torch = nn.MSELoss()
    
    computed_loss = loss.compute(target, output)
    computed_loss_torch = loss_torch(output_torch, target)
    
    assert torch.allclose(computed_loss, computed_loss_torch), "Test Same Loss FAILED"
    
    
    
    grad_loss = loss.compute_gradient(output, target)
    
    grad_loss_torch, = torch.autograd.grad(computed_loss_torch, output_torch, retain_graph = True)
    
    assert torch.allclose(grad_loss, grad_loss_torch), "Test Same Grad Loss FAILED"
    
    
    
    grad_all, = seq.backward(grad_loss)
    
    grad_all_torch, = torch.autograd.grad(computed_loss_torch, input_, retain_graph = True)
    
    assert torch.allclose(grad_all, grad_all_torch), "Test Same Overall Grad FAILED"
    
    
    
    grad_weight1 = lin_layer1.grad_weight
    grad_weight2 = lin_layer2.grad_weight

    grad_weight_torch1, = torch.autograd.grad(computed_loss_torch, lin_layer_torch1.weight, retain_graph = True)
    grad_weight_torch2, = torch.autograd.grad(computed_loss_torch, lin_layer_torch2.weight, retain_graph = True)

    assert torch.allclose(grad_weight1, grad_weight_torch1) and torch.allclose(grad_weight2, grad_weight_torch2), "Test Same Weights Grad Lin FAILED"



    grad_bias1 = lin_layer1.grad_bias.T
    grad_bias2 = lin_layer2.grad_bias.T

    grad_bias_torch1, = torch.autograd.grad(computed_loss_torch, lin_layer_torch1.bias, retain_graph = True)
    grad_bias_torch2, = torch.autograd.grad(computed_loss_torch, lin_layer_torch2.bias, retain_graph = True)

    assert torch.allclose(grad_bias1, grad_bias_torch1) and torch.allclose(grad_bias2, grad_bias_torch2), "Test Same Bias Grad Lin FAILED"
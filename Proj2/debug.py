# -*- coding: utf-8 -*-

import math
import torch
from torch import empty, tensor, nn

from framework.models import Sequential
from framework.modules import Linear, Relu, Tanh
from framework.losses import MSE

train_input = tensor([0, 1, 2, 3]).float().reshape(-1,1)
train_input.requires_grad = True
train_target = tensor([0, 1]).float().reshape(-1,1)

test_lin_layer = Linear(4, 2)
test_lin_layer_torch = nn.Linear(4, 2)

test_lin_layer.weight = test_lin_layer_torch.weight.clone().detach()
test_lin_layer.weight.requires_grad = False

test_lin_layer.bias = test_lin_layer_torch.bias.clone().detach().reshape(-1, 1)
test_lin_layer.bias.requires_grad = False

output, = test_lin_layer(train_input)
output_torch = test_lin_layer_torch(train_input.T).T
output_torch.requires_grad = True

assert torch.all(output == output_torch)

loss = MSE()
loss_torch = nn.MSELoss()

computed_loss = loss.compute(output, train_target)
computed_loss_torch = loss_torch(output_torch, train_target)
computed_loss_torch.requires_grad = True

assert computed_loss == computed_loss_torch

grad_loss = loss.compute_gradient(output, train_target)

computed_loss_torch.backward()
grad_loss_torch = output_torch.grad

print(grad_loss, "\n", grad_loss_torch)

test_lin_layer.backward(loss.compute_gradient(output, train_target))

# -*- coding: utf-8 -*-

import math
from torch import empty, set_grad_enabled
set_grad_enabled(False)

from framework.models import Sequential
from framework.modules import Linear, Relu, Tanh
from framework.losses import MSE


def generate_data(n_training_samples = 1000, n_test_samples = 1000):
    all_inputs = empty(n_training_samples + n_test_samples, 2).uniform_()

    all_classes = ((all_inputs - 0.5).pow(2).sum(axis = 1).sqrt() < 1 / math.sqrt(2*math.pi)).long()
    all_target  = empty(n_training_samples + n_test_samples, 2).zero_()
    all_target[all_classes == 0, 0] = 1
    all_target[all_classes == 1, 1] = 1
    print(all_classes)
    
    train_input , test_input  = all_inputs.split([n_training_samples, n_test_samples])
    train_target, test_target = all_target.split([n_training_samples, n_test_samples])

    import matplotlib.pyplot as plt
    import torch
    
    print(train_target)
    print(test_target.shape)
    print(train_input.shape)
    print(test_input.shape)
    assert torch.all(train_target[:,0] == torch.logical_not(train_target[:,1]))
    assert torch.all(test_target[:,0] == torch.logical_not(test_target[:,1]))
    assert torch.all(torch.logical_or(test_target == 1, test_target == 0))
    
    plt.scatter(train_input[:,0], train_input[:,1], c = torch.argmax(train_target, axis = 1))
    plt.show()
    plt.scatter(test_input[:,0], test_input[:,1], c = torch.argmax(test_target, axis = 1))
    plt.show()


    return train_input, train_target, test_input, test_target


def random_permutation(n):
    array = [i for i in range(n)]
    
    random_permutation_array = []
    while array:
        idx = empty(1).random_(0, len(array)).int()
        random_permutation_array.append( array[idx] )
        array.pop(idx)
        
    return random_permutation_array



train_input, train_target, test_input, test_target = generate_data()

model = Sequential(Linear(2 , 25, weight_initializer = "he_normal", bias_initializer = "zeros"), 
                   Relu(),
                   Linear(25, 25, weight_initializer = "he_normal", bias_initializer = "zeros"),
                   Relu(),
                   Linear(25, 25, weight_initializer = "he_normal", bias_initializer = "zeros"),
                   Relu(),
                   Linear(25, 2 , weight_initializer = "xavier_normal", bias_initializer = "zeros"),
                   Tanh())
loss = MSE()




n_epochs = 20
for epoch in range(n_epochs):
    random_idx = random_permutation(len(train_target))
    
    train_losses = []
    for idx in random_idx:
        sample_input  = train_input [idx].reshape(-1, 1)
        sample_target = train_target[idx].reshape(-1, 1)
        
        prediction, = model(sample_input)
        loss_sample = loss.compute(sample_target, prediction)
        train_losses.append(loss_sample)
        
        grad_loss = loss.compute_gradient(sample_target, prediction)
        model.backward(grad_loss)
        model.update_params(100)

    print("Epoch {}. Train Loss: {}".format(epoch + 1, sum(train_losses) / len(train_losses)))
        
        
        
        
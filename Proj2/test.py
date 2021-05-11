# -*- coding: utf-8 -*-

import math
from torch import empty, set_grad_enabled
set_grad_enabled(False)

from framework.models import Sequential
from framework.modules import Linear, Relu, Tanh
from framework.losses import MSE
from framework.callbacks import EarlyStopping

import matplotlib.pyplot as plt


def generate_data(n_training_samples = 1000, n_test_samples = 1000):
    all_inputs = empty(n_training_samples + n_test_samples, 2).uniform_()

    all_classes = ((all_inputs - 0.5).pow(2).sum(axis = 1).sqrt() < 1 / math.sqrt(2*math.pi)).long()
    all_target  = empty(n_training_samples + n_test_samples, 2).zero_()
    all_target[all_classes == 0, 0] = 1
    all_target[all_classes == 1, 1] = 1
    
    train_input , test_input  = all_inputs.split([n_training_samples, n_test_samples])
    train_target, test_target = all_target.split([n_training_samples, n_test_samples])

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
criterion = MSE()

n_epochs = 50
lr = 1e-3
early_stopping = EarlyStopping(patience = 20)


train_losses   = []
train_accuracy = []
test_losses    = []
test_accuracy  = []

for epoch in range(n_epochs):
    random_idx = random_permutation(len(train_target))
    
    train_losses_epoch   = []
    train_accuracy_epoch = []
    
    for idx in random_idx:
        sample_input  = train_input [idx].reshape(-1, 1)
        sample_target = train_target[idx].reshape(-1, 1)
        
        prediction, = model(sample_input)
                
        loss = criterion.compute(sample_target, prediction)
        train_losses_epoch.append(loss)
        train_accuracy_epoch.append(prediction.argmax() == sample_target.argmax())
        
        grad_loss = criterion.compute_gradient(sample_target, prediction)
        model.backward(grad_loss)
        
        model.update_params(lr)
        
    
    test_losses_epoch   = []
    test_accuracy_epoch = []
    
    for idx in range(len(test_target)):
        sample_input  = test_input [idx].reshape(-1, 1)
        sample_target = test_target[idx].reshape(-1, 1)
        
        prediction, = model(sample_input)
        
        loss = criterion.compute(sample_target, prediction)
        test_losses_epoch.append(loss)
        test_accuracy_epoch.append(prediction.argmax() == sample_target.argmax())
        
        
    train_losses  .append( sum(train_losses_epoch  ) / len(train_losses_epoch) )
    train_accuracy.append( sum(train_accuracy_epoch) / len(train_accuracy_epoch) )
    test_losses   .append( sum(test_losses_epoch   ) / len(test_losses_epoch) )
    test_accuracy .append( sum(test_accuracy_epoch ) / len(test_accuracy_epoch) )
    
    print("Epoch {}:\n    Train Loss    : {:.5g}\n    Train Accuracy: {:.5g}\n    Test Loss     : {:.5g}\n    Test Accuracy : {:.5g}\n".format(epoch + 1, train_losses[-1], train_accuracy[-1], test_losses[-1], test_accuracy[-1]))
    
    if early_stopping(model, test_losses[-1]):
        break
    
    
    
    
    
    
    
    
    
    
    
    
def plot_history(train_losses, train_accuracy, test_losses, test_accuracy):
        
    plt.plot(train_losses, label = 'Train')
    plt.plot(test_losses, label = 'Test')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
    plt.plot(train_accuracy, label = 'Train')
    plt.plot(test_accuracy, label = 'Test')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
        
    
import numpy as np    
from torch import FloatTensor
import matplotlib.patches as mpatches

def plot_decision_boundary(model, xlim = (0, 1), ylim = (0, 1), xstep = 1e-3, ystep = 1e-3):

    # DO NOT TRY TO REDO WITHOUT NUMPY ETC!!!!! THIS MUST BE REMOVED AD PRIORI BECAUSE THERE IS NO WAY
    # OF CONVERTING LIST TO TENSOR WITH ONLY TORCH.EMPTY    
    
    xlist = []
    ylist = []
    for axis_lim, axis_step, axis_list in zip([xlim, ylim], [xstep, ystep], [xlist, ylist]):
        start, end    = axis_lim
        n_points_axis = math.ceil((end - start) / axis_step)
                
        axis_list.extend( [p * (end - start) / n_points_axis + start for p in range(n_points_axis + 1)] )
    
    XX, YY = np.meshgrid(xlist, ylist) 
    meshgrid = [(x, y) for x in xlist for y in ylist]
    
    preds = [model(FloatTensor(point).reshape(-1, 1))[0].argmax() for point in meshgrid]
    
    preds = np.reshape(preds, XX.shape)
    
    
    fig, ax = plt.subplots(1, 1, figsize = (5, 5))
    ax.contourf(XX, YY, preds, levels = 1, colors = ['white', 'red'])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_xlabel("First Tensor Dimention", size = 14)
    ax.set_ylabel("Second Tensor Dimention", size = 14)
    
    
    targets = [((FloatTensor(point) - 0.5).pow(2).sum().sqrt() < 1 / math.sqrt(2*math.pi)).long() for point in meshgrid]
    targets = np.reshape(targets, XX.shape)
    ax.contourf(XX, YY, targets, levels = 1, colors = ['white', 'blue'], alpha = 0.5)
    
    red_patch  = mpatches.Patch(color='red', label='Learnt Decision Region')
    blue_patch = mpatches.Patch(color='blue', label='Ground Truth')

    ax.legend(handles = [red_patch, blue_patch], prop = {'size': 12})
    ax.set_title("Comparison of Learnt Decision Region\nagainst Ground Truth", fontsize = 14, fontweight = 'bold')
                 
    

plot_history(train_losses, train_accuracy, test_losses, test_accuracy)
plot_decision_boundary(model)
# -*- coding: utf-8 -*-

import math
from torch import empty, set_grad_enabled
set_grad_enabled(False)

from framework.models import Sequential
from framework.layers import Linear, ReLU, Tanh
from framework.losses import LossMSE
from framework.callbacks import EarlyStopping
from framework.learning_rate_schedules import TimeDecayLR

import matplotlib.pyplot as plt


def generate_data(n_training_samples = 1000, n_test_samples = 1000):
    """Generates desired numbers of samples for training and test set by uniformly
    sampling them in [0,1] x [0,1] and assigning a label 0 to those outside the
    disk centered at (0.5, 0.5) of radius 1/sqrt(2*pi) and label 1 to the others.
    
    Args:
        n_training_samples::[int]
            Number of samples to generate for the training set.
        n_test_samples::[int]
            Number of samples to generate for the test set.
    Returns:
        train_input::[torch.Tensor]
            Tensor of shape (n_training_samples, 2) containing generated points
            for training set input.
        train_target::[torch.Tensor]
            Tensor of shape (n_training_samples, 2) containing labels in one-hot
            format corresponding to train_input.
        test_input::[torch.Tensor]
            Tensor of shape (n_training_samples, 2) containing generated points
            for test set input.
        test_target::[torch.Tensor]
            Tensor of shape (n_training_samples, 2) containing labels in one-hot
            format corresponding to test_input.
    """
    all_inputs = empty(n_training_samples + n_test_samples, 2).uniform_()

    all_target = ((all_inputs - 0.5).pow(2).sum(axis = 1).sqrt() < 1 / math.sqrt(2*math.pi)).long()
    
    train_input , test_input  = all_inputs.split([n_training_samples, n_test_samples])
    train_target, test_target = all_target.split([n_training_samples, n_test_samples])

    return train_input, train_target, test_input, test_target


def random_permutation(n):
    """Creates a random permutation of natural numbers going from 0 (inclusive) 
    to n (exclusive).
    
    Args:
        n::[int]
            Largest value (exclusive) of randomly permuted values in range(0, n).
    Returns:
        random_permutation_array::[list]
            Randomly ordered values in range(0, n).
    """
    array = list(range(n))
    
    random_permutation_array = []
    while array:
        idx = empty(1).random_(0, len(array)).int()
        random_permutation_array.append( array[idx] )
        array.pop(idx)
        
    return random_permutation_array


def train_or_predict_epoch(model, inputs, targets, criterion, training, lr = None):
    """Function evaluating all samples in inputs using model, and computes 
    mean loss (using criterion) and accuracy over all the samples. If training
    flag is True, the samples are parsed in random order and backpropagation
    and parameter update of the model using SGD is performed with learning rate
    lr. If training flag is True, lr must be provided.

    Args:
        model::[Module]
            Instance of Module used to generate predictions for each sample in 
            input, and if training flag is True, whose parameters are updated
            via SGD.
        inputs::[torch.Tensor]
            Tensor of shape (n_samples, 2) containing points to be used during 
            training or prediction with the model.
        targets::[torch.Tensor]
            Tensor of shape (n_samples, 2) containing labels in one-hot
            format corresponding to inputs.
        criterion::[Loss]
            Instance of Loss used to calcolate loss of each sample from target
            and model predictions.
        training::[boolean]
            Boolean flag determining if inputs samples should be parsed in random
            order and if backpropagation and parameters update should be performed.
        lr::[float]
            Value to use as learning rate during parameters update. If training
            is False, value is ignored.
    Returns:
        mean_loss::[float]
            Mean loss of model predictions computed using all samples in inputs and
            their respective target.
        mean_accuracy::[float]
            Mean accuracy of model predictions computed using all samples in inputs and
            their respective target.
    """
    indices = random_permutation(len(targets)) if training else range(len(targets))
    
    losses     = []
    accuracies = []
    
    for idx in indices:
        sample_input  = inputs [idx].reshape(-1, 1)
        sample_target = targets[idx].reshape(-1, 1)
        
        prediction, = model(sample_input)
        predicted_class = (prediction > 0.5).long()
                
        loss = criterion(sample_target, prediction)
        losses    .append(loss)
        accuracies.append((predicted_class == sample_target).squeeze())
        
        if training:
            assert lr is not None, \
            "No learning rate provided to train_or_predict_epoch despite" + \
            " parameter training = True."
            
            grad_loss = criterion.backward(sample_target, prediction)
            model.backward(grad_loss)
            model.update_params(lr)
            
    mean_loss     = sum(losses)     / len(losses)        
    mean_accuracy = sum(accuracies) / len(accuracies)        

    return mean_loss, mean_accuracy



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


train_input, train_target, test_input, test_target = generate_data()

model = Sequential(Linear(2 , 25, weight_initializer = "he_normal", bias_initializer = "zeros"), 
                   ReLU(),
                   Linear(25, 25, weight_initializer = "he_normal", bias_initializer = "zeros"),
                   ReLU(),
                   Linear(25, 25, weight_initializer = "he_normal", bias_initializer = "zeros"),
                   ReLU(),
                   Linear(25, 1 , weight_initializer = "xavier_normal", bias_initializer = "zeros"),
                   Tanh())

criterion = LossMSE()

n_epochs = 500
lr_scheduler = TimeDecayLR(initial_lr = 0.1, decay = 0.7)
early_stopping = EarlyStopping(patience = 20, verbose = False)

train_losses   = []
train_accuracy = []
test_losses    = []
test_accuracy  = []

for epoch in range(n_epochs):
    # Calculate learning rate for current epoch.
    lr = lr_scheduler()
    
    # Training on Training set.
    train_loss_epoch, train_accuracy_epoch = train_or_predict_epoch(model, 
                                                                    train_input, 
                                                                    train_target, 
                                                                    criterion, 
                                                                    training = True, 
                                                                    lr = lr)
    
    # Evaluating current model on Test set.
    test_loss_epoch , test_accuracy_epoch  = train_or_predict_epoch(model, 
                                                                    test_input,
                                                                    test_target,
                                                                    criterion, 
                                                                    training = False)
      
    # Storing  and logging losses and accuracies for current epoch.
    train_losses  .append( train_loss_epoch )
    train_accuracy.append( train_accuracy_epoch )
    test_losses   .append( test_loss_epoch )
    test_accuracy .append( test_accuracy_epoch )
    
    print("Epoch {}:".format(epoch + 1))
    print("    Train Loss    : {:.5g}".format(train_loss_epoch))
    print("    Train Accuracy: {:.5g}".format(train_accuracy_epoch))
    print("    Test Loss     : {:.5g}".format(test_loss_epoch))
    print("    Test Accuracy : {:.5g}".format(test_accuracy_epoch), end = "\n\n")
    
    # Testing if we should stop training.
    if early_stopping(model, test_loss_epoch):
        break

final_train_loss, final_train_accuracy = train_or_predict_epoch(model, 
                                                                train_input, 
                                                                train_target, 
                                                                criterion, 
                                                                training = False)
final_test_loss , final_test_accuracy  = train_or_predict_epoch(model, 
                                                                test_input, 
                                                                test_target, 
                                                                criterion, 
                                                                training = False)

# Logging losses and accuracy at end of training.
print("Final Model:")
print("    Train Loss    : {:.5g}".format(final_train_loss))
print("    Train Accuracy: {:.5g}".format(final_train_accuracy))
print("    Test Loss     : {:.5g}".format(final_test_loss))
print("    Test Accuracy : {:.5g}".format(final_test_accuracy))

plot_history(train_losses, train_accuracy, test_losses, test_accuracy)
    
    

    
        
    
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
    
    preds = [model(FloatTensor(point).reshape(-1, 1))[0] > 0.5 for point in meshgrid]
    
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

    ax.legend(handles = [red_patch, blue_patch], prop = {'size': 12}, loc='upper center', bbox_to_anchor=(0.45, -0.15), ncol = 2)
    ax.set_title("Comparison of Learnt Decision Region\nagainst Ground Truth", fontsize = 14, fontweight = 'bold')
                 
    

plot_history(train_losses, train_accuracy, test_losses, test_accuracy)
plot_decision_boundary(model)
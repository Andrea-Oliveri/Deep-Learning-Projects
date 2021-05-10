import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import matplotlib.pyplot as plt

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def compute_nb_errors(model, data_input, data_target):    
    
    output= model(data_input)
    #ConvNetAux outputs two values, the first one corresponds to the target prediction
    if type(output)==tuple : output=output[0]
    predicted_classes = output.argmax(axis=1)
    nb_data_errors = (predicted_classes != data_target).sum()

    return nb_data_errors.item()


def train_model_aux(model,\
    train_input, train_target, train_classes,\
    test_input, test_target, test_classes,\
    nb_epochs =50, mini_batch_size=100, lr=1e-3, beta=0.5):
    '''
    Trains a model which accounts for auxiliary loss.
    Returns : loss_train, loss_test, errors_train, errors_test in percent
    '''
    criterion_digits     = nn.CrossEntropyLoss()
    criterion_comparison = nn.CrossEntropyLoss()
    optimizer            = optim.Adam(model.parameters(), lr = lr)

    # 2 is number of losses to track: digit, comparison
    loss_train = torch.zeros(nb_epochs, 2, requires_grad = False)
    loss_test  = torch.zeros(nb_epochs, 2, requires_grad = False)
    
    # 2 is number of accuracies to track: digit, comparison
    errors_train = torch.zeros(nb_epochs, 2, requires_grad = False)
    errors_test  = torch.zeros(nb_epochs, 2, requires_grad = False)

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            batch_input   = train_input  .narrow(0, b, mini_batch_size)
            batch_target  = train_target .narrow(0, b, mini_batch_size)
            batch_classes = train_classes.narrow(0, b, mini_batch_size).view(-1)
            
            output_comparison, digits_pred = model(batch_input)
                        
            loss_digits     = criterion_digits(digits_pred, batch_classes)
            loss_comparison = criterion_comparison(output_comparison, batch_target)
            loss_tot        = beta * loss_comparison + (1 - beta) * loss_digits
                        
            optimizer.zero_grad()
            loss_tot .backward()
            optimizer.step()
            
            loss_train[e] += torch.tensor([loss_digits, loss_comparison]) # This normalization factor is probably wrong. Double check
                        
            errors_digits     = (digits_pred.argmax(axis = 1) != batch_classes).sum()/(train_input.size(0)*2)*100 # *2 to account that we have 2*batch_size digits
            errors_comparison = (output_comparison.argmax(axis = 1) != batch_target).sum()/train_input.size(0)*100

            errors_train[e] += torch.tensor([errors_digits, errors_comparison])
            
            
        with torch.no_grad():
            model.eval()
            for b in range(0, test_input.size(0), mini_batch_size):
                batch_input   = test_input  .narrow(0, b, mini_batch_size)
                batch_target  = test_target .narrow(0, b, mini_batch_size)
                batch_classes = test_classes.narrow(0, b, mini_batch_size).view(-1)

                output_comparison, digits_pred = model(batch_input)

                loss_digits     = criterion_digits(digits_pred, batch_classes)
                loss_comparison = criterion_comparison(output_comparison, batch_target)
                loss_tot        = beta * loss_digits + (1 - beta) * loss_comparison

                loss_test[e] += torch.tensor([loss_digits, loss_comparison])
                
                errors_digits     = (digits_pred.argmax(axis = 1) != batch_classes).sum()/(train_input.size(0)*2)*100
                errors_comparison = (output_comparison.argmax(axis = 1) != batch_target).sum()/(train_input.size(0))*100

                errors_test[e] += torch.tensor([errors_digits, errors_comparison])

            model.train()

    return loss_train, loss_test, errors_train, errors_test

def train_model(model,\
    train_input, train_target,\
    test_input, test_target,\
    nb_epochs =50, mini_batch_size=100, lr=1e-3):
    '''
    Trains a model which accounts for auxiliary loss.
    Returns : loss_train, loss_test, errors_train, errors_test in percent
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    loss_train = torch.zeros(nb_epochs, requires_grad = False).cuda()
    loss_test  = torch.zeros(nb_epochs, requires_grad = False).cuda()
    
    errors_train = torch.zeros(nb_epochs, requires_grad = False).cuda()
    errors_test  = torch.zeros(nb_epochs, requires_grad = False).cuda()

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            batch_input   = train_input  .narrow(0, b, mini_batch_size)
            batch_target  = train_target .narrow(0, b, mini_batch_size)
            
            output = model(batch_input)                        
            loss   = criterion(output, batch_target)
                        
            optimizer.zero_grad()
            loss     .backward()
            optimizer.step()
            
            loss_train[e] += loss.item() # This normalization factor is probably wrong. Double check
                        
            errors= (output.argmax(axis = 1) != batch_target).sum()/train_input.size(0)*100

            errors_train[e] += errors
            
            
        with torch.no_grad():
            model.eval()
            for b in range(0, test_input.size(0), mini_batch_size):
                batch_input   = test_input  .narrow(0, b, mini_batch_size)
                batch_target  = test_target .narrow(0, b, mini_batch_size)

                output = model(batch_input)                        
                loss   = criterion(output, batch_target)

                loss_test[e] += loss.item() # This normalization factor is probably wrong. Double check
                            
                errors = (output.argmax(axis = 1) != batch_target).sum()/train_input.size(0)*100

                errors_test[e] += errors

            model.train()

    return loss_train.unsqueeze(1), loss_test.unsqueeze(1), errors_train.unsqueeze(1), errors_test.unsqueeze(1)

def plot_loss(loss_train, loss_test, errors_train, errors_test):

    fig, (ax_loss, ax_errors) = plt.subplots(1, 2, figsize = (15, 4))
    ax_loss.plot(loss_train[:,0].cpu().detach())
    ax_loss.plot(loss_test[:,0] .cpu().detach())
    if loss_train.size(1)==2:
        ax_loss.plot(loss_train[:,1].cpu().detach())
        ax_loss.plot(loss_test[:,1] .cpu().detach())
    ax_loss.legend(['loss train digit','loss test digit', \
                    'loss train comparison','loss test comparison'])
    ax_loss.set_xlabel('# epochs')

    ax_errors.plot(errors_train[:,0].cpu().detach())
    ax_errors.plot(errors_test[:,0] .cpu().detach())
    if errors_train.size(1)==2:
        ax_errors.plot(errors_train[:,1].cpu().detach())
        ax_errors.plot(errors_test[:,1] .cpu().detach())
    ax_errors.legend(['errors train digit','errors test digit',\
                       'errors train comparison', 'errors test comparison'])
    ax_errors.set_ylabel('% errors')
    ax_errors.set_xlabel('# epochs')
    plt.show()


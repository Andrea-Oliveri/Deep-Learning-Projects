import torch
from torch import nn
from torch import optim

from callbacks import EarlyStopping
from utils import count_nb_parameters
import dlc_practical_prologue as prologue


def train_or_predict_epoch(model, inputs, targets,
                           criterion_comparison, 
                           mini_batch_size, training, optimizer = None, 
                           use_auxiliary_loss = False,
                           classes = None, criterion_digits = None,
                           beta = None):
    """Function evaluating all samples in inputs using model, and computing
    average loss (using criterion_comparison) and accuracy over all the samples. 
    If training flag is True, the samples are parsed in random order and 
    backpropagation is performed as well as a parameter update of the model 
    using the provided optimizer. Otherwise, the prediction is run temporarely
    disabling autograd. If use_auxiliary_loss is set to True, then classes, beta 
    and criterion_digits must also be provided, and will be used to compute an 
    additional loss for the predicted digits, in addition to the prediction
    of the final comparison. In this case, the model should return the predictions
    for the digits as well as the final prediction.

    Args:
        model::[torch.nn.Module]
            Instance of Module used to generate predictions for each sample in 
            input, and if training flag is True, whose parameters are updated
            using Adam.
        inputs::[torch.Tensor]
            Tensor of shape (n_samples, 2, 14, 14) in which each sample is
            an image with two channels, where each channel is an mnist digit.
        targets::[torch.Tensor]
            Tensor of shape (n_samples, ) containing labels in corresponding 
            to inputs.
        classes::[torch.Tensor]
            Tensor of shape (n_samples, 2) containing the digit to which
            each channel of inputs belongs.
        criterion_comparison::[torch.nn.Loss]
            Instance of torch.nn.Loss used to calculate loss of each sample from
            target and model predictions for comparison.
        criterion_digit::[torch.nn.Loss]
            Instance of torch.nn.Loss used to calculate loss of each sample from
            classes and model predictions for digit.
        training::[boolean]
            Boolean flag determining if inputs samples should be parsed in random
            order and if backpropagation and parameters update should be performed.
        optimizer::[torch.optim.Optimizer]
            Optimizer to use for parameters update. If training is False, this
            parameter is ignored.
        use_auxiliary_loss::[boolean]
            Boolean flag determining if model is using auxiliary loss.
        beta::[float]
            Coefficient used when training models using auxiliary loss to 
            weight the importance of the comparison and digit loss in total
            loss. loss_tot = beta * loss_comparison + (1. - beta) * loss_digits
    Returns:
        mean_loss::[torch.tensor]
            Mean loss of model predictions computed using all samples in inputs and
            their respective target (as well as mean loss corresponding to digits
            predictions if use_auxiliary_loss is True).
        mean_accuracy::[torch.tensor]
            Mean accuracy of model predictions computed using all samples in inputs and
            their respective target (as well as mean accuracy corresponding to digits
            predictions if use_auxiliary_loss is True).
    """
    indices = torch.randperm(len(targets)) if training else range(len(targets))
    
    if use_auxiliary_loss:
        for param, param_name in zip([classes, criterion_digits, beta],
                                     ["classes", "criterion_digits", "beta"]):
            assert param is not None, \
                   f"No {param_name} parameter provided to train_or_predict_epoch " + \
                    "despite parameter use_auxiliary_loss = True."
            
        inputs, targets, classes = inputs[indices], targets[indices], classes[indices]
    else:
        inputs, targets = inputs[indices], targets[indices]

    losses = []
    accuracies = []
    
    for b in range(0, inputs.size(0), mini_batch_size):
        batch_input   = inputs .narrow(0, b, mini_batch_size)
        batch_target  = targets.narrow(0, b, mini_batch_size)
        
        if use_auxiliary_loss:
            batch_classes = classes.narrow(0, b, mini_batch_size).view(-1)
            output_comparison, digits_pred = model(batch_input)
            
        else:
            output_comparison = model(batch_input)


        loss_comparison = criterion_comparison(output_comparison, batch_target)
        
        if use_auxiliary_loss:
            loss_digits     = criterion_digits(digits_pred, batch_classes)
            loss_tot        = beta * loss_comparison + (1. - beta) * loss_digits
        else:
            loss_tot = loss_comparison

        if training:
            assert optimizer is not None, \
                   "No optimizer provided to train_or_predict_epoch despite " + \
                   "parameter training = True."
            optimizer.zero_grad()
            loss_tot.backward()
            optimizer.step()

        accuracy_comparison = (output_comparison.argmax(axis = 1) == batch_target ).float().mean()    

        if use_auxiliary_loss:
            accuracy_digits = (digits_pred.argmax(axis = 1) == batch_classes).float().mean()
            losses.append( (loss_comparison, loss_digits, loss_tot) )  
            accuracies.append( (accuracy_comparison, accuracy_digits) )
        else:
            losses.append( (loss_comparison, ) )  
            accuracies.append( (accuracy_comparison, ) )

    return torch.tensor(losses).mean(axis = 0), torch.tensor(accuracies).mean(axis = 0)



def train_model(model, train_input, train_target, test_input, test_target,
                nb_epochs = 50, mini_batch_size = 100, lr = 1e-3,
                patience = 20, use_auxiliary_loss = False, train_classes = None,
                test_classes = None, beta = None, early_stop_verbose = False):
    """
    Performs the full training of the model with given train and test input and
    targets. If use_auxiliary_loss is set to True, then train_classes,
    test_classes and beta must also be provided, and will be used to compute
    an additional loss for the predicted digits, in addition to the prediction
    of the final comparison. In this case, the model should return the predictions
    for the digits as well as the final prediction.
    
    Args:
        model::[torch.nn.Module]
            Instance of Module used to generate predictions for each sample in 
            input, and if training flag is True, whose parameters are updated
            using Adam.
        train_input::[torch.Tensor]
            Tensor of shape (n_samples, 2, 14, 14) in which each sample is
            an image with two channels, where each channel is an mnist digit.
        train_target::[torch.Tensor]
            Tensor of shape (n_samples, ) containing labels in corresponding 
            to train_input.
        train_classes::[torch.Tensor]
            Tensor of shape (n_samples, 2) containing the digit to which
            each channel of train_input belongs.
        test_input::[torch.Tensor]
            Tensor of shape (n_samples, 2, 14, 14) in which each sample is
            an image with two channels, where each channel is an mnist digit.
        test_target::[torch.Tensor]
            Tensor of shape (n_samples, ) containing labels in corresponding 
            to test_input.
        test_classes::[torch.Tensor]
            Tensor of shape (n_samples, 2) containing the digit to which
            each channel of test_input belongs.
        nb_epochs::[int]
            Number of epochs for which to train the model.
        mini_batch_size::[int]
            Batch size to be used during training.
        lr::[float]
            Learning rate used in the Adam optimizer.
        patience::[int]
            Patience value used by EarlyStopping.
        verbose::[int]
            Set the verbose level of EarlyStopping.
        use_auxiliary_loss::[boolean]
            Boolean flag determining if model is using auxiliary loss.
        beta::[float]
            Coefficient used when training models using auxiliary loss to 
            weight the importance of the comparison and digit loss in total
            loss. loss_tot = beta * loss_comparison + (1. - beta) * loss_digits
    Returns : 
        train_loss::[torch.Tensor]
            Tensor of length nb_epochs containing the comparison (and digits
            loss as well, if use_auxiliary_loss is True) loss computed on the 
            training set for each epoch.
        test_loss::[torch.Tensor]
            Tensor of length nb_epochs containing the comparison (and digits
            loss as well, if use_auxiliary_loss is True) loss computed on the 
            test set for each epoch.
        train_accuracy::[torch.Tensor]
            Tensor of length nb_epochs containing the comparison (and digits
            loss as well, if use_auxiliary_loss is True) accuracy computed on the 
            training set for each epoch. 
        test_accuracy::[torch.Tensor]
            Tensor of length nb_epochs containing the comparison (and digits
            loss as well, if use_auxiliary_loss is True) accuracy computed on the 
            test set for each epoch.
        final_weights_epoch::[int]
            Epoch number to which the final model weights correspond to. 
            Needed to index test_accuracy and get the accuracy 
            corresponding to epoch restored by early stopping.
    """
    
    if use_auxiliary_loss:
        for param, param_name in zip([train_classes, test_classes, beta],
                                     ["train_classes", "test_classes", "beta"]):
            assert param is not None, \
                   f"No {param_name} parameter provided to train_or_predict_epoch " + \
                    "despite parameter use_auxiliary_loss = True."
    
        
    criterion_digits = nn.CrossEntropyLoss() if use_auxiliary_loss else None
    
    criterion_comparison = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    early_stopping = EarlyStopping(patience = patience, verbose = early_stop_verbose)
    
    train_loss = []
    test_loss  = []
    
    train_accuracy = []
    test_accuracy  = []
    
    for epoch in range(nb_epochs):       

        # If use_auxiliary_loss is false, parameters classes and criterion_digits
        # will be ignored and can therefore be left to None. 
        train_loss_epoch, train_accuracy_epoch = \
            train_or_predict_epoch(model = model, 
                                   inputs = train_input, 
                                   targets = train_target,
                                   classes = train_classes,
                                   criterion_comparison = criterion_comparison,
                                   criterion_digits = criterion_digits,
                                   beta = beta,
                                   mini_batch_size = mini_batch_size,
                                   training = True,
                                   optimizer = optimizer,
                                   use_auxiliary_loss = use_auxiliary_loss)
            
        with torch.no_grad():
            model.eval()
            
            # If use_auxiliary_loss is false, parameters classes and criterion_digits
            # will be ignored and can therefore be left to None.
            test_loss_epoch, test_accuracy_epoch = \
                train_or_predict_epoch(model = model, 
                                       inputs = test_input, 
                                       targets = test_target,
                                       classes = test_classes,
                                       criterion_comparison = criterion_comparison,
                                       criterion_digits = criterion_digits,
                                       beta = beta,
                                       mini_batch_size = mini_batch_size,
                                       training = False,
                                       use_auxiliary_loss = use_auxiliary_loss)
            
            model.train()
        
        train_loss.append(train_loss_epoch)
        test_loss .append(test_loss_epoch)
        train_accuracy.append(train_accuracy_epoch)
        test_accuracy .append(test_accuracy_epoch)
            
        # Early Stopping uses the total loss (combination of digits prediction and 
        # comparison predictions) to decide.
        if early_stopping(model, test_loss_epoch[-1]):
            final_weights_epoch = epoch - patience
            break
        else:
            final_weights_epoch = epoch
        
    return {'train_loss'    : torch.stack(train_loss)    .requires_grad_(False),
            'test_loss'     : torch.stack(test_loss )    .requires_grad_(False),
            'train_accuracy': torch.stack(train_accuracy).requires_grad_(False),
            'test_accuracy' : torch.stack(test_accuracy) .requires_grad_(False),
            'final_weights_epoch': final_weights_epoch}



def train_multiple_times(model_creating_func, parameters, model_name):
    """
    Trains a model with a certain set of parameters for a given number of 
    repetitions.

    Args:
        model_creating_func::[function]
            Function to be used to instanciate the model.
        parameters::[dict]
            Dictionary containing the relevant parameters for model creation,
            training (number of epochs, learning rate, mini batch size, ...), 
            the number of runs which must be performed with this model and 
            the number of samples to generate in the training and test set
            used for each run.
        model_name::[str]
            The name of the model which will be trained. Only needed to print
            on terminal.

    Returns:
        results::[list]
            List containing dictionaries wih keys 'train_loss', 'test_loss',
            'train_accuracy', 'test_accuracy', 'final_weights_epoch', containing
            measures of these metrics for each epoch, for each run (with the
            exception of final_weights_epoch which is just an int per run).
    """
    print(f"Collecting measures for {model_name} model")

    n_repetitions     = parameters.pop('n_repetitions')
    n_samples_dataset = parameters.pop('n_samples_dataset')
    
    params_training = {k: v for k, v in parameters.items() if k in ["beta", "lr", "mini_batch_size", "nb_epochs", "early_stop_verbose", "use_auxiliary_loss"]}
    
    params_model    = {k: v for k, v in parameters.items() if k not in params_training}
    
    
    print("Model has {} parameters to train" \
          .format(count_nb_parameters(model_creating_func(**params_model))))
    
    results = []
    
    for i in range(n_repetitions):
        print(f"Performing measure {i+1} of {n_repetitions}")
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(n_samples_dataset)

        model = model_creating_func(**params_model)
        
        results_repetition  = train_model(model = model, 
                                          train_input = train_input, 
                                          train_target = train_target, 
                                          train_classes = train_classes, 
                                          test_input = test_input,
                                          test_target = test_target,
                                          test_classes = test_classes,
                                          **params_training)
        
        results.append( results_repetition )
        
    print('\n')
    
    return results
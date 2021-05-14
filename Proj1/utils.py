import torch
from torch import nn
from torch import optim

import matplotlib.pyplot as plt
from callbacks import EarlyStopping



def train_or_predict_epoch(model, inputs, targets, 
                           criterion_comparison, 
                           mini_batch_size, training, optimizer = None, 
                           use_auxiliary_loss = False,
                           classes = None, criterion_digits = None,
                           beta = None):
    """Function evaluating all samples in inputs using model, and computes 
    mean loss (using criterion) and accuracy over all the samples. If training
    flag is True, the samples are parsed in random order and backpropagation
    and parameter update of the model using SGD is performed with learning rate
    lr. If training flag is True, lr must be provided.

    Args:
        model::[torch.nn.Module]
            Instance of Module used to generate predictions for each sample in 
            input, and if training flag is True, whose parameters are updated
            via SGD.
        inputs::[torch.Tensor]
            Tensor of shape (n_samples, 2) containing points to be used during 
            training or prediction with the model.
        targets::[torch.Tensor]
            Tensor of shape (n_samples, 2) containing labels in one-hot
            format corresponding to inputs.
        criterion::[torch.nn._Loss]
            Instance of torch.nn._Loss used to calcolate loss of each sample from
            target and model predictions.
        training::[boolean]
            Boolean flag determining if inputs samples should be parsed in random
            order and if backpropagation and parameters update should be performed.
        optimizer::[torch.optim.Optimizer]
            Optimizer to use for parameters update. If training is False, this
            parameter is ignored.
    Returns:
        mean_loss::[float]
            Mean loss of model predictions computed using all samples in inputs and
            their respective target.
        mean_accuracy::[float]
            Mean accuracy of model predictions computed using all samples in inputs and
            their respective target.
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
            losses.append( (loss_digits, loss_comparison, loss_tot) )  
            accuracies.append( (accuracy_digits, accuracy_comparison) )
        else:
            losses.append( (loss_comparison, ) )  
            accuracies.append( (accuracy_comparison, ) )

    return torch.tensor(losses).mean(axis = 0), torch.tensor(accuracies).mean(axis = 0)





def train_model(
    model,
    train_input,
    train_target,
    test_input,
    test_target,
    nb_epochs=50,
    mini_batch_size=100,
    lr=1e-3,
    patience = 20,
    use_auxiliary_loss = False,
    train_classes = None,
    test_classes = None,
    beta=None,
    verbose = False
):
    """
    Trains a model which accounts for auxiliary loss.
    Returns : 
        loss_train::[Tensor]
            Tensor of shape [nb_epochs, 1, 1] containing  loss_comparison, loss_digits for each epoch of the training
        loss_test::[Tensor]
            Tensor of shape [nb_epochs, 1, 1] containing loss_comparison, loss_digits for each epoch of testing
        errors_train::[Tensor]
            Tensor of shape [nb_epochs, 1, 1] containing errors_comparison, errors_digits for each epoch of the training
        errors_test
            Tensor of shape [nb_epochs, 1, 1] containing errors_comparison, errors_digits for each epoch of testing
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
    early_stopping = EarlyStopping(patience = patience, verbose = verbose)
    
    train_loss = []
    test_loss  = []
    
    train_accuracy = []
    test_accuracy  = []
    
    for _ in range(nb_epochs):       

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
            break

    return {'train_loss'    : torch.stack(train_loss)    .requires_grad_(False),
            'test_loss'     : torch.stack(test_loss )    .requires_grad_(False),
            'train_accuracy': torch.stack(train_accuracy).requires_grad_(False),
            'test_accuracy' : torch.stack(test_accuracy) .requires_grad_(False) }



def plot_loss(loss_train, loss_test, errors_train, errors_test):
    fig, (ax_loss, ax_errors) = plt.subplots(1, 2, figsize=(15, 4))
    ax_loss.plot(loss_train[5:, 0].cpu().detach())
    ax_loss.plot(loss_test[5:, 0].cpu().detach())
    #If loss_digits exists
    if loss_train.size(1) == 2:
        ax_loss.plot(loss_train[5:, 1].cpu().detach())
        ax_loss.plot(loss_test[5:, 1].cpu().detach())
    ax_loss.legend(
        [
            "loss train comparison",
            "loss test comparison",
            "loss train digit",
            "loss test digit",

        ]
    )
    ax_loss.set_xlabel("# epochs")

    ax_errors.plot(errors_train[:, 0].cpu().detach())
    ax_errors.plot(errors_test[:, 0].cpu().detach())
    #If error_digits exists
    if errors_train.size(1) == 2:
        ax_errors.plot(errors_train[:, 1].cpu().detach())
        ax_errors.plot(errors_test[:, 1].cpu().detach())
    ax_errors.legend(
        [
            "accuracy train comparison",
            "accuracy test comparison",
            "accuracy train digit",
            "accuracy test digit",
        ]
    )
    ax_errors.set_ylabel("accuracy")
    ax_errors.set_xlabel("# epochs")
    
    plt.draw()


def count_nb_parameters(model):
    nb_parameters = sum([param.numel() for param in model.parameters()])

    return nb_parameters    


def plot_and_analyse_results(results):

    metrics = results[0].keys()
    grouped_metrics = {metric: [res[metric] for res in results] for metric in metrics}

    early_stop_epoch = [torch.argmin(test_loss) for test_loss in grouped_metrics['test_loss']]
    mean_early_stop_epoch = sum(early_stop_epoch) / len(early_stop_epoch)

    # Draw metrics for each run individually.
    fig, axes = plt.subplots(2, 2, figsize = (10, 5))
        
    for ax, metric in zip(axes.ravel(), metrics):
        metric_name = metric.replace('_', ' ').title()
        ax.set_xlabel("Epoch Number")
        ax.set_ylabel(metric_name)
        
        for metric_training in grouped_metrics[metric]:
            ax.plot(metric_training)

        ax.vlines(mean_early_stop_epoch, *ax.get_ylim(), linestyle = '--', color = 'red')

    
    plt.tight_layout()
    plt.show()
    
    # Draw average metrics over all runs.
    fig, axes = plt.subplots(2, 2, figsize = (10, 5))
        
    for ax, metric in zip(axes.ravel(), metrics):
        metric_name = metric.replace('_', ' ').title()
        ax.set_xlabel("Epoch Number")
        ax.set_ylabel(metric_name)
        
        min_nb_epochs = min([len(m) for m in grouped_metrics[metric]])
        metric_same_nb_epochs = [m[:min_nb_epochs] for m in grouped_metrics[metric]]
        mean_metric = torch.stack(metric_same_nb_epochs).mean(axis = 0)
        ax.plot(mean_metric)
        ax.vlines(mean_early_stop_epoch, *ax.get_ylim(), linestyle = '--', color = 'red')

    
    plt.tight_layout()
    plt.show()


    # Collect Statistics
    #metrics_at_early_stop = early_stop_epoch

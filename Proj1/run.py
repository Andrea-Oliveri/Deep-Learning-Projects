import torch
import dlc_practical_prologue as prologue

from models import MLP, MLPAux, ConvNet, ConvNetAux
from utils import train_model, count_nb_parameters, plot_and_analyse_results




def train_multiple_times(model_creating_func, parameters):
    print(f"Collecting Measures for {model_creating_func.__name__} Model")

    n_repetitions     = parameters.pop('n_repetitions')
    n_samples_dataset = parameters.pop('n_samples_dataset')
    
    params_training = {k: v for k, v in parameters.items() if k in ["beta", "lr", "mini_batch_size", "nb_epochs", "verbose", "use_auxiliary_loss"]}
    
    params_model    = {k: v for k, v in parameters.items() if k not in params_training}
    
    
    print("Model has {} parameters to train".format(count_nb_parameters(model_creating_func(**params_model))))
    
    results = []
    
    for i in range(n_repetitions):
        print(f"Performing Measure {i+1} of {n_repetitions}", end = "\r")
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
        
    return results


        



n_repetitions = 20
n_samples_dataset = 1000
        


results = train_multiple_times(MLP, 
                               {'n_repetitions': n_repetitions,
                                'n_samples_dataset': n_samples_dataset,
                                'nb_epochs': 200, 
                                'mini_batch_size': 20,
                                'lr': 1e-3,
                                'use_auxiliary_loss': False,
                                'verbose': False,
                                'p': 0.4,
                                'nb_hidden1': 70,
                                'nb_hidden2': 20})
    
plot_and_analyse_results(results)

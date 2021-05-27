from models import *
from utils import print_results_summary
from training import train_multiple_times

n_repetitions = 20
n_samples_dataset = 1000

models_to_run = {
    
        "Fully Connected Net (not siamese, no auxiliary loss)":
                { 'model_creating_func': FullyConnectedNet,
                  'n_repetitions': n_repetitions,
                  'n_samples_dataset': n_samples_dataset,
                  'nb_epochs': 200, 
                  'mini_batch_size': 20,
                  'lr': 1e-3,
                  'use_auxiliary_loss': False,
                  'p': 0.5,
                  'nb_hidden1': 100,
                  'nb_hidden2': 50,
                  'nb_hidden3': 10}, 
                
                
        # Model which performs same treatement on both channels of input image
        # (weight sharing) but does not train with the auxiliary loss is
        # implemented exactly the same as the model which does the weight
        # sharing and uses the auxiliary loss, but the beta parameter is set
        # to 1, hence making impact of auxiliary loss on total loss and on
        # backpropagated gradients zero.
        "Fully Connected Net (siamese, no auxiliary loss)":
                { 'model_creating_func': FullyConnectedNetAux,
                  'n_repetitions': n_repetitions,
                  'n_samples_dataset': n_samples_dataset,
                  'nb_epochs': 200, 
                  'mini_batch_size': 20,
                  'lr': 1e-3,
                  'use_auxiliary_loss': True,
                  'beta': 1.0, 
                  'p': 0.5,
                  'nb_hidden1': 100,
                  'nb_hidden2': 10}, 
                
                
        "Fully Connected Net (siamese, auxiliary loss)":
                { 'model_creating_func': FullyConnectedNetAux,
                  'n_repetitions': n_repetitions,
                  'n_samples_dataset': n_samples_dataset,
                  'nb_epochs': 200, 
                  'mini_batch_size': 20,
                  'lr': 1e-3,
                  'use_auxiliary_loss': True,
                  'beta': 0.5, 
                  'p': 0.5,
                  'nb_hidden1': 100,
                  'nb_hidden2': 40},
                
                
        "Convolutional Net (not siamese, no auxiliary loss)":
                { 'model_creating_func': ConvolutionalNet,
                  'n_repetitions': n_repetitions,
                  'n_samples_dataset': n_samples_dataset,
                  'nb_epochs': 500, 
                  'mini_batch_size': 20,
                  'lr': 1e-3,
                  'use_auxiliary_loss': False,
                  'p': 0.6,
                  'nb_channel1': 32,
                  'nb_channel2': 64,
                  'nb_hidden1': 20,
                  'nb_hidden2': 10,
                  'padding': 0,
                  'k_size': 4},
                
                
        # Model which performs same treatement on both channels of input image
        # (weight sharing) but does not train with the auxiliary loss is
        # implemented exactly the same as the model which does the weight
        # sharing and uses the auxiliary loss, but the beta parameter is set
        # to 1, hence making impact of auxiliary loss on total loss and on
        # backpropagated gradients zero.
        "Convolutional Net (siamese, no auxiliary loss)":
                { 'model_creating_func': ConvolutionalNetAux,
                  'n_repetitions': n_repetitions,
                  'n_samples_dataset': n_samples_dataset,
                  'nb_epochs': 500, 
                  'mini_batch_size': 20,
                  'lr': 1e-3,
                  'use_auxiliary_loss': True,
                  'beta': 1.0,
                  'p': 0.6,
                  'nb_channel1': 32,
                  'nb_channel2': 64,
                  'nb_hidden': 40,
                  'padding': 2,
                  'k_size': 5},
                
                
        "Convolutional Net (siamese, auxiliary loss)":
                { 'model_creating_func': ConvolutionalNetAux,
                  'n_repetitions': n_repetitions,
                  'n_samples_dataset': n_samples_dataset,
                  'nb_epochs': 500, 
                  'mini_batch_size': 20,
                  'lr': 1e-3,
                  'use_auxiliary_loss': True,
                  'beta': 0.7,
                  'p': 0.6,
                  'nb_channel1': 32,
                  'nb_channel2': 64,
                  'nb_hidden': 40,
                  'padding': 2,
                  'k_size': 5}
        }


for model_name, model_params in models_to_run.items():
    model_creating_func = model_params.pop('model_creating_func')
    
    results = train_multiple_times(model_creating_func, model_params, model_name)
    
    print_results_summary(results, model_name)
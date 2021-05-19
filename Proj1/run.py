import torch

from models import MLP, MLPAux, ConvNet, ConvNetAux
from utils import results_summary
from trainers import train_multiple_times
from plotting import plot_results

n_repetitions = 20
n_samples_dataset = 1000
        
MLP_results = train_multiple_times(MLP, 
                               {'n_repetitions': n_repetitions,
                                'n_samples_dataset': n_samples_dataset,
                                'nb_epochs': 200, 
                                'mini_batch_size': 20,
                                'lr': 1e-3,
                                'use_auxiliary_loss': False,
                                'verbose': False,
                                'p': 0.5,
                                'nb_hidden1': 100,
                                'nb_hidden2': 20})


MLPAux_results = train_multiple_times(MLPAux, 
                               {'n_repetitions': n_repetitions,
                                'n_samples_dataset': n_samples_dataset,
                                'nb_epochs': 200, 
                                'mini_batch_size': 50,
                                'lr': 1e-3,
                                'use_auxiliary_loss': True,
                                'verbose': False,
                                'p': 0.2,
                                'nb_hidden': 100,
                                'beta': 0.7})

ConvNet_results = train_multiple_times(ConvNet, 
                               {'n_repetitions': n_repetitions,
                                'n_samples_dataset': n_samples_dataset,
                                'nb_epochs': 500, 
                                'mini_batch_size': 10,
                                'lr': 1e-3,
                                'use_auxiliary_loss': False,
                                'verbose': False,
                                'p': 0.6,
                                'nb_channel1': 64,
                                'nb_channel2': 64,
                                'nb_hidden': 20,
                                'beta': 0.3,
                                'padding': 0,
                                'k_size': 4}
                                )

ConvNetAux_results = train_multiple_times(ConvNetAux, 
                               {'n_repetitions': n_repetitions,
                                'n_samples_dataset': n_samples_dataset,
                                'nb_epochs': 500, 
                                'mini_batch_size': 200,
                                'lr': 1e-3,
                                'use_auxiliary_loss': True,
                                'verbose': False,
                                'p': 0.6,
                                'nb_channel1': 32,
                                'nb_channel2': 64,
                                'beta': 0.3,
                                'padding': 1,
                                'k_size': 5}
                                )
                                
MLP_summary=results_summary(MLP_results,use_auxiliary_loss=False)
MLPAux_summary=results_summary(MLPAux_results,use_auxiliary_loss=True)
ConvNet_summary=results_summary(ConvNet_results,use_auxiliary_loss=False)
ConvNetAux_summary=results_summary(ConvNetAux_results,use_auxiliary_loss=True)

print('\n\n')
print(f'\n After {n_repetitions} repetitions of the MLP model the performance is :\
        \n Mean comparison accuracy: {MLP_summary[0]}\
        \n STD comparison accuracy: {MLP_summary[1]}\
        \n Min comparison accuracy: {MLP_summary[2]}\
        \n Max comparison accuracy: {MLP_summary[3]}')

print(f'\n After {n_repetitions} repetitions of the MLPAux model the performance is :\
        \n Mean comparison accuracy: {MLPAux_summary[0]}\
        \n STD comparison accuracy: {MLPAux_summary[1]}\
        \n Min comparison accuracy: {MLPAux_summary[4]}\
        \n Max comparison accuracy: {MLPAux_summary[5]}\
        \n Mean digit accuracy: {MLPAux_summary[2]}\
        \n STD digit accuracy: {MLPAux_summary[3]}\
        \n Min digit accuracy: {MLPAux_summary[6]}\
        \n Max digit accuracy: {MLPAux_summary[7]}')

print(f'\n After {n_repetitions} repetitions of the ConvNet model the performance is :\
        \n Mean comparison accuracy: {ConvNet_summary[0]}\
        \n STD comparison accuracy: {ConvNet_summary[1]}\
        \n Min comparison accuracy: {ConvNet_summary[2]}\
        \n Max comparison accuracy: {ConvNet_summary[3]}')

print(f'\n After {n_repetitions} repetitions of the ConvNetAux model the performance is :\
        \n Mean comparison accuracy: {ConvNetAux_summary[0]}\
        \n STD comparison accuracy: {ConvNetAux_summary[1]}\
        \n Min comparison accuracy: {ConvNetAux_summary[4]}\
        \n Max comparison accuracy: {ConvNetAux_summary[5]}\
        \n Mean digit accuracy: {ConvNetAux_summary[2]}\
        \n STD digit accuracy: {ConvNetAux_summary[3]}\
        \n Min digit accuracy: {ConvNetAux_summary[6]}\
        \n Max digit accuracy: {ConvNetAux_summary[7]}')

plot_results(MLP_results,use_auxiliary_loss=False)
plot_results(MLPAux_results,use_auxiliary_loss=True)
plot_results(ConvNet_results,use_auxiliary_loss=False)
plot_results(ConvNetAux_results,use_auxiliary_loss=True)
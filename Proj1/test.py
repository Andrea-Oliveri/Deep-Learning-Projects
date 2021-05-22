import torch

from models import MLP, MLPAux, ConvNet, ConvNetAux
from utils import results_summary
from trainers import train_multiple_times
from plotting import plot_results

n_repetitions = 20
n_samples_dataset = 1000
models=[MLP, MLPAux, ConvNet, ConvNetAux]
params={'MLP': {        'n_repetitions': n_repetitions,
                        'n_samples_dataset': n_samples_dataset,
                        'nb_epochs': 200, 
                        'mini_batch_size': 20,
                        'lr': 1e-3,
                        'use_auxiliary_loss': False,
                        'verbose': False,
                        'p': 0.5,
                        'nb_hidden1': 100,
                        'nb_hidden2': 20}, 

        'MLPAux': {     'n_repetitions': n_repetitions,
                        'n_samples_dataset': n_samples_dataset,
                        'nb_epochs': 200, 
                        'mini_batch_size': 100,
                        'lr': 1e-3,
                        'use_auxiliary_loss': True,
                        'verbose': False,
                        'p': 0.2,
                        'nb_hidden': 100,
                        'beta': 0.8}, 

        'ConvNet':{     'n_repetitions': n_repetitions,
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
                        'k_size': 4},

        'ConvNetAux':{  'n_repetitions': n_repetitions,
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
        }


for model in models:
        auxiliary_loss=model.__name__.endswith('Aux')
        results=train_multiple_times(model, params[model.__name__])
        summary=results_summary(results,use_auxiliary_loss=auxiliary_loss)

        print('Writing results summary')
        with open(model.__name__+'_summary.log','w') as f:
                f.write(f' After {n_repetitions} repetitions of the '+ model.__name__ +f' model, the performance is :\
                        \n Mean comparison accuracy: {summary[0]:0.3f} \
                        \n STD comparison accuracy: {summary[1]:0.3f} \
                        \n Min comparison accuracy: {summary[2]:0.3f} \
                        \n Max comparison accuracy: {summary[3]:0.3f}')
                if auxiliary_loss:
                        f.write(f'\n Mean digit accuracy: {summary[4]:0.3f}\
                                  \n STD digit accuracy: {summary[5]:0.3f}\
                                  \n Min digit accuracy: {summary[6]:0.3f}\
                                  \n Max digit accuracy: {summary[7]:0.3f}')

        print('Saving plots of the runs')
        if auxiliary_loss:
                fig_comparison, fig_digit=plot_results(results=results, use_auxiliary_loss=auxiliary_loss)
                fig_comparison.savefig(model.__name__+'_comparison.png',dpi=400)
                fig_digit.savefig(model.__name__+'_digit.png',dpi=400)
        else:
                fig_comparison=plot_results(results=results, use_auxiliary_loss=auxiliary_loss)
                fig_comparison.savefig(model.__name__+'_comparison.png',dpi=400)

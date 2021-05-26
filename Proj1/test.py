from models import *
from utils import results_summary
from training import train_multiple_times
from plotting import plot_results

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
                  'verbose': False,
                  'p': 0.5,
                  'nb_hidden1': 100,
                  'nb_hidden2': 20,
                  'nb_hidden3': 20}, 
                
        # Siamese but no auxiliary loss implementes as exactly the same as
        # siamese + auxiliary loss, but beta is set to 1 so auxiliary loss
        # has no impact during back-propagation.
        "Fully Connected Net (siamese, no auxiliary loss)":
                { 'model_creating_func': FullyConnectedNetAux,
                  'n_repetitions': n_repetitions,
                  'n_samples_dataset': n_samples_dataset,
                  'nb_epochs': 200, 
                  'mini_batch_size': 20,
                  'lr': 1e-3,
                  'use_auxiliary_loss': True,
                  'beta': 1.0, 
                  'verbose': False,
                  'p': 0.5,
                  'nb_hidden1': 100,
                  'nb_hidden2': 20}, 
                
        "Fully Connected Net (siamese, auxiliary loss)":
                { 'model_creating_func': FullyConnectedNetAux,
                  'n_repetitions': n_repetitions,
                  'n_samples_dataset': n_samples_dataset,
                  'nb_epochs': 200, 
                  'mini_batch_size': 20,
                  'lr': 1e-3,
                  'use_auxiliary_loss': True,
                  'beta': 0.5, 
                  'verbose': False,
                  'p': 0.5,
                  'nb_hidden1': 100,
                  'nb_hidden2': 20},
                
        "Convolutional Net (not siamese, no auxiliary loss)":
                { 'model_creating_func': ConvolutionalNet,
                  'n_repetitions': n_repetitions,
                  'n_samples_dataset': n_samples_dataset,
                  'nb_epochs': 500, 
                  'mini_batch_size': 20,
                  'lr': 1e-3,
                  'use_auxiliary_loss': False,
                  'verbose': False,
                  'p': 0.6,
                  'nb_channel1': 32,
                  'nb_channel2': 64,
                  'nb_hidden1': 20,
                  'nb_hidden2': 20,
                  'padding': 0,
                  'k_size': 4},
                
        
        "Convolutional Net (siamese, no auxiliary loss)":
                { 'model_creating_func': ConvolutionalNetAux,
                  'n_repetitions': n_repetitions,
                  'n_samples_dataset': n_samples_dataset,
                  'nb_epochs': 500, 
                  'mini_batch_size': 20,
                  'lr': 1e-3,
                  'use_auxiliary_loss': True,
                  'beta': 1.0,
                  'verbose': False,
                  'p': 0.6,
                  'nb_channel1': 32,
                  'nb_channel2': 64,
                  'nb_hidden': 20,
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
                  'verbose': False,
                  'p': 0.6,
                  'nb_channel1': 32,
                  'nb_channel2': 64,
                  'nb_hidden': 20,
                  'padding': 2,
                  'k_size': 5}
        }


for model_name, model_params in models_to_run.items():
    model_creating_func = model_params.pop('model_creating_func')
    results = train_multiple_times(model_creating_func, model_params, model_name)
    summary = results_summary(results, use_auxiliary_loss = model_params['use_auxiliary_loss'])

    with open("./results/" + model_name+'_summary.log','w') as f:
            f.write(f"{model_name}\n{[res['test_accuracy'][res['final_weights_epoch']] for res in results]}\n" + \
                    f"\n Mean comparison accuracy: {summary[0]:0.3f} \
                    \n STD comparison accuracy: {summary[1]:0.3f} \
                    \n Min comparison accuracy: {summary[2]:0.3f} \
                    \n Max comparison accuracy: {summary[3]:0.3f}")
            if model_params['use_auxiliary_loss']:
                    f.write(f'\n Mean digit accuracy: {summary[4]:0.3f}\
                              \n STD digit accuracy: {summary[5]:0.3f}\
                              \n Min digit accuracy: {summary[6]:0.3f}\
                              \n Max digit accuracy: {summary[7]:0.3f}')

    if model_params['use_auxiliary_loss']:
            fig_comparison, fig_digit=plot_results(results=results, use_auxiliary_loss = model_params['use_auxiliary_loss'], title = model_name)
            fig_comparison.savefig("./results/" +model_name+'_comparison.pdf', bbox_inches='tight')
            fig_digit.savefig("./results/" +model_name+'_digit.pdf', bbox_inches='tight')
    else:
            fig_comparison=plot_results(results=results, use_auxiliary_loss = model_params['use_auxiliary_loss'], title = model_name)
            fig_comparison.savefig("./results/" +model_name+'_comparison.pdf', bbox_inches='tight')

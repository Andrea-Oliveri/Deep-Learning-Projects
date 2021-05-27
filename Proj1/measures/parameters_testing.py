import torch
import dlc_practical_prologue as prologue
import warnings
import pickle
import os 

from models import *
from utils import *
from training import *




def train_all_params(model_creating_func, params_dict, pickle_path):
    all_params_combinations = []
    
    warnings.filterwarnings("ignore")
    
    dtypes = [torch.tensor(v).dtype for v in params_dict.values()]
    params_linspaces = [torch.tensor(v).float() for v in params_dict.values()]
    
    params_mesh = [t.flatten().type(dtypes[i]) for i, t in enumerate(torch.meshgrid(*params_linspaces))]
    
    warnings.filterwarnings("default")
    
    print("Testing {} Combinations".format(len(params_mesh[0])))
    
    
    if os.path.isfile(pickle_path):
        with open(pickle_path, "rb") as file:
            already_done = pickle.load(file)
        measures = already_done
    else:
        measures=[]
    
    count=0

    for params in zip(*params_mesh):
        count += 1
        print("\n\nStarting Training {} of {}".format(count, len(params_mesh[0])))
        p_dict = {k: v.item() for k, v in zip(params_dict.keys(), params)}
        print("Parameters:", p_dict)

        if p_dict in [m[0] for m in measures]:
            continue

        train_input, train_target, train_classes, test_input, test_target, test_classes = [x.cuda() for x in prologue.generate_pair_sets(1000)]
        
        params_training = {k: v for k, v in p_dict.items() if k in ["beta", "lr", "mini_batch_size", "nb_epochs"]}
        params_training["use_auxiliary_loss"] = "beta" in params_training

        params_model = {k: v for k, v in p_dict.items() if k not in params_training}
        
        try:
            model = model_creating_func(**params_model).cuda()

            results  = train_model(model = model, 
                                   train_input = train_input, 
                                   train_target = train_target, 
                                   train_classes = train_classes, 
                                   test_input = test_input,
                                   test_target = test_target,
                                   test_classes = test_classes,
                                   **params_training)
            measures.append( (dict(p_dict), results) )
            
            # with open("ConvNetAux_measures_updated.pkl", "wb") as file:
            #     pickle.dump(measures, file)
        except Exception as e:
            if 'Output size is too small' in str(e):
                print(e)
            else:
                raise e
        
    return measures


pickle_path = "FullyConnectedNet.pkl"
measures = train_all_params(FullyConnectedNet, {
                  'nb_epochs': 200, 
                  'mini_batch_size': 20,
                  'lr': 1e-3,
                  'use_auxiliary_loss': False,
                  'p': 0.5,
                  'nb_hidden1': 100,
                  'nb_hidden2': [20, 50, 100],
                  'nb_hidden3': [10, 20, 40]}
                  ,pickle_path) # Adam's default. Should adapt it anyway.
                           
with open(pickle_path, "wb") as file:
       pickle.dump(measures, file)

pickle_path = "FullyConnectedNetAux_noAux.pkl"
measures = train_all_params(FullyConnectedNetAux, {
                  'nb_epochs': 200, 
                  'mini_batch_size': 20,
                  'lr': 1e-3,
                  'use_auxiliary_loss': True,
                  'beta': 1.0, 
                  'p': 0.5,
                  'nb_hidden1': [100],
                  'nb_hidden2': [10, 20, 40]},
                  pickle_path)

with open(pickle_path, "wb") as file:
    pickle.dump(measures, file)

pickle_path = "FullyConnectedNetAux_Aux.pkl"
measures = train_all_params(FullyConnectedNetAux, {
                  'nb_epochs': 200, 
                  'mini_batch_size': 20,
                  'lr': 1e-3,
                  'use_auxiliary_loss': True,
                  'beta': [0.5, 0.7], 
                  'p': 0.5,
                  'nb_hidden1': [20, 40, 100],
                  'nb_hidden2': [10, 20, 40]},
                  pickle_path)

with open(pickle_path, "wb") as file:
    pickle.dump(measures, file)
        
pickle_path = "ConvolutionalNet.pkl"
measures = train_all_params(ConvolutionalNet, {
                  'nb_epochs': 500, 
                  'mini_batch_size': 20,
                  'lr': 1e-3,
                  'use_auxiliary_loss': False,
                  'p': 0.6,
                  'nb_channel1': 32,
                  'nb_channel2': 64,
                  'nb_hidden1': [10, 20, 40],
                  'nb_hidden2': [10, 20, 40],
                  'padding': 0,
                  'k_size': 4},
                  pickle_path)

with open(pickle_path, "wb") as file:
    pickle.dump(measures, file)

pickle_path = "ConvolutionalNetAux_noAux.pkl"
measures = train_all_params(ConvolutionalNetAux, {
                  'nb_epochs': 500, 
                  'mini_batch_size': 20,
                  'lr': 1e-3,
                  'use_auxiliary_loss': True,
                  'beta': 1.0,
                  'p': 0.6,
                  'nb_channel1': 32,
                  'nb_channel2': 64,
                  'nb_hidden': [10, 20, 40],
                  'padding': 2,
                  'k_size': 5},
                  pickle_path)

with open(pickle_path, "wb") as file:
    pickle.dump(measures, file)

pickle_path = "ConvolutionalNetAux_Aux.pkl"
measures = train_all_params(ConvolutionalNetAux, {
                  'nb_epochs': 500, 
                  'mini_batch_size': 20,
                  'lr': 1e-3,
                  'use_auxiliary_loss': True,
                  'beta': [0.1, 0.2, 0.5, 0.7, 0.9],
                  'p': 0.6,
                  'nb_channel1': 32,
                  'nb_channel2': 64,
                  'nb_hidden': [10, 20, 40],
                  'padding': 2,
                  'k_size': 5},
                  pickle_path)

with open(pickle_path, "wb") as file:
    pickle.dump(measures, file)
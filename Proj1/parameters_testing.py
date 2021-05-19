import torch
import dlc_practical_prologue as prologue
import warnings
import pickle
import os 

from models import *
from utils import *




def train_all_params(model_creating_func, params_dict):
    all_params_combinations = []
    
    warnings.filterwarnings("ignore")
    
    dtypes = [torch.tensor(v).dtype for v in params_dict.values()]
    params_linspaces = [torch.tensor(v).float() for v in params_dict.values()]
    
    params_mesh = [t.flatten().type(dtypes[i]) for i, t in enumerate(torch.meshgrid(*params_linspaces))]
    
    warnings.filterwarnings("default")
    
    print("Testing {} Combinations".format(len(params_mesh[0])))
    
    
    
    with open("ConvNetAux_measures_updated.pkl", "rb") as file:
        already_done = pickle.load(file)
    
    
    measures = already_done
    count = len(measures)
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
            
            with open("ConvNetAux_measures_updated.pkl", "wb") as file:
                pickle.dump(measures, file)
        except Exception as e:
            if 'Output size is too small' in str(e):
                print(e)
            else:
                raise e
        
    return measures


        
        
        
redo_cached = False




# Fully connected net with 2 hidden layers. Channels are concatenated into the
# same vector at input.
# pickle_path = "MLP_measures.pkl"
# if True or redo_cached or not os.path.exists(pickle_path):

#    measures = train_all_params(MLP, {'p': [0.4,0.5],
#                                      'nb_hidden1': [100],
#                                      'nb_hidden2': [20, 30],
#                                      'nb_epochs': [200],
#                                      'mini_batch_size': [10,20,50],
#                                      'lr': [0.001,0.0001]}) # Adam's default. Should adapt it anyway.
                           
#    with open(pickle_path, "wb") as file:
#        pickle.dump(measures, file)
        
        
        
# Fully connected net with 2 hidden layers. Channels undergo the same treatement
# (they are concatenated along batch dimention). Second-last Linear layer predicts digits.
# When beta = 1, the auxiliary loss is ignored -> Only final loss counts. May want to perform 
# comparison to see how useful auxiliary loss is over just treating channels separately.
# pickle_path = "MLPAux_measures.pkl"
# if True or redo_cached or not os.path.exists(pickle_path):

#     measures = train_all_params(MLPAux, {'p': [0.2],
#                                          'nb_hidden': [100],
#                                          'nb_epochs': [400],
#                                          'mini_batch_size': [10,100],
#                                          'lr': [0.001,0.0001], # Adam's default. Should adapt it anyway.
#                                          'beta': [0.7,0.8,0.9]})
                           
#     with open(pickle_path, "wb") as file:
#         pickle.dump(measures, file)
        
        
# pickle_path = "ConvNet_measures_updated.pkl"
# if True or redo_cached or not os.path.exists(pickle_path):
    # measures = train_all_params(ConvNet, {'p': [0.6],
    #                                 'nb_hidden': [20, 40],
    #                                 'k_size': [4],
    #                                 'padding': [1], # 0 or 1
    #                                 'nb_channel_1': [32, 64],
    #                                 'nb_channel_2': [64],
    #                                 'nb_epochs': [500],
    #                                 'mini_batch_size': [10],
    #                                 'lr': [0.001]}) # Adam's default. Should adapt it anyway.
                        
with open(pickle_path, "wb") as file:
    pickle.dump(measures, file)
        
pickle_path = "ConvNetAux_measures_updated.pkl"
if True or redo_cached or not os.path.exists(pickle_path):

    measures = train_all_params(ConvNetAux, {'p': [0.6],
                                      'k_size': [5],
                                      'padding': [1], # 0 or 1
                                      'nb_channel1': [32],
                                      'nb_channel2': [64],
                                      'nb_epochs': [501],
                                      'mini_batch_size': [200],
                                      'lr': [0.001],# Adam's default. Should adapt it anyway.
                                      'beta': [0.7]}) 
                           
    with open(pickle_path, "wb") as file:
        pickle.dump(measures, file)
        

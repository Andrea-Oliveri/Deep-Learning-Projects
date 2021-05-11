import torch
import dlc_practical_prologue as prologue

from models import *
from trainings import *
from utils import *

train_input, train_target, train_classes, test_input, test_target, test_classes = [x.cuda() for x in prologue.generate_pair_sets(1000)]

for model in [MLP,MLPAux,ConvNet,ConvNetAux]:
    #TODO Create real parameters lists
    print(f'Computing parameters sets for model : {model.__name__} ...')
    ps=[0.5]
    nb_hiddens=[100]
    betas=[0.2]
    nb_epochs=[100,150]
    mini_batch_sizes=[100]
    lrs=[1e-3]

    errors_train_comparison, errors_test_comparison, errors_train_digits, errors_test_digits = parameters_training(model, train_input, train_target, train_classes, test_input, test_target, test_classes, ps, nb_hiddens,betas,mini_batch_sizes,lrs,nb_epochs)
    

    print(f'The five best parameters set are for {model.__name__} are : \n{errors_test_comparison[errors_test_comparison[:,0].argsort()][:5]}')
plt.show()
from utils import weight_reset, train_model, train_model_aux, plot_loss
import torch


def parameters_training(model_empty, train_input, train_target, train_classes, test_input, test_target, test_classes,\
                        ps, nb_hiddens, betas, mini_batch_sizes, lrs, nb_epochs ):
    '''
    Args: 
        ps::[list]
            List of dropout probabilities
        nb_hidden::[list]
            List of the number of hidden units in the first FC layer
        betas::[list]
            List of betas that act as balance between loss_digit and loss_comparison : loss_tot = beta * loss_comparison + (1 - beta) * loss_digits
        mini_batch_sizes::[list]
            List of mini_batch_sizes to use during training epochs
        nb_epochs::[list]
            List of the total number of epochs used during training

    Returns :
        errors_train_digits::[Tensor]
            Tensor containing [errors_train_digits_final, p, nb_hidden, beta, mini_batch_size, lrs, nb_epoch ] for each set of parameters
        errors_test_digits::[Tensor]
            Tensor containing [errors_test_digits_final, p, nb_hidden, beta, mini_batch_size, lrs, nb_epoch ] for each set of parameters
        errors_train_comparison::[Tensor] 
            Tensor containing [errors_train_comparison_final, p, nb_hidden, beta, mini_batch_size, lrs, nb_epoch ] for each set of parameters
        errors_test_comparison::[Tensor]
            Tensor containing [errors_test_comparison_final, p, nb_hidden, beta, mini_batch_size, lrs, nb_epoch ] for each set of parameters
    '''

    errors_train_digits      =[]
    errors_train_comparison  =[]
    errors_test_digits       =[]
    errors_test_comparison   =[]
    
    for p in ps:
        for nb_hidden in nb_hiddens:
            for mini_batch_size in mini_batch_sizes:
                for lr in lrs:
                    for nb_epoch in nb_epochs:
                        if model_empty.__name__.endswith('Aux'):
                            for beta in betas :
                                #Reset model parameters
                                model=model_empty(p=p,nb_hidden=nb_hidden).cuda()
                                space=''
                                print(f'Model parameters : drop probability : {p}')
                                print(f'{space:>19}nb_hidden = {nb_hidden}')
                                print(f'Computing with: mini batch size = {mini_batch_size}')
                                print(f'{space:>16}learning rate = {lr}')
                                print(f'{space:>16}number of epochs = {nb_epoch}')
                                print(f'{space:>16}beta = {beta}')
                                # Shuffling
                                # perm=torch.randperm(train_input.size(0))
                                # train_input=train_input[perm]
                                # train_target=train_target[perm]
                                # test_input=test_input[perm]
                                # test_target=test_target[perm]
                                
                                loss_train, loss_test, errors_train, errors_test=train_model_aux(model, train_input, train_target, train_classes, test_input, test_target, test_classes, beta=beta, lr=lr, mini_batch_size=mini_batch_size, nb_epochs=nb_epoch)

                                plot_loss(loss_train, loss_test, errors_train, errors_test)
                                
                                # Adds final errors future comparison
                                errors_train_digits.append([errors_train[-1,0].item(), p, nb_hidden, mini_batch_size, lr, nb_epoch, beta])
                                errors_train_comparison.append([errors_train[-1,1].item(), p, nb_hidden, mini_batch_size, lr, nb_epoch, beta])
                                errors_test_digits.append([errors_test[-1,0].item(), p, nb_hidden, mini_batch_size, lr, nb_epoch, beta])
                                errors_test_comparison.append([errors_test[-1,1].item(), p, nb_hidden, mini_batch_size, lr, nb_epoch, beta])
                                
                                print(f'Final error train digit: {errors_train[-1,0]:0.2f}')
                                print(f'Final error train comparison: {errors_train[-1,1]:0.2f}')
                                print(f'Final error test digit: {errors_test[-1,0]:0.2f}')
                                print(f'Final error test comparison: {errors_test[-1,1]:0.2f}')
                                print('\n\n')

                        else:
                            #Reset model parameters
                            model=model_empty(p=p,nb_hidden=nb_hidden).cuda()
                            space=''
                            print(f'Model parameters : drop probability : {p}')
                            print(f'{space:>19}nb_hidden = {nb_hidden}')
                            print(f'Computing with: mini batch size = {mini_batch_size}')
                            print(f'{space:>16}learning rate = {lr}')
                            print(f'{space:>16}number of epochs = {nb_epoch}')
                            #Shuffling
                            # perm=torch.randperm(train_input.size(0))
                            # train_input=train_input[perm]
                            # train_target=train_target[perm]
                            # test_input=test_input[perm]
                            # test_target=test_target[perm]

                            loss_train, loss_test, errors_train, errors_test=train_model(model, train_input, train_target, test_input, test_target,lr=lr, mini_batch_size=mini_batch_size, nb_epochs=nb_epoch)
                            
                            plot_loss(loss_train, loss_test, errors_train, errors_test)

                            # Adds final errors for future comparison
                            errors_train_comparison.append([errors_train[-1,0].item(), p, nb_hidden, mini_batch_size, lr, nb_epoch])
                            errors_test_comparison.append([errors_test[-1,0].item(), p, nb_hidden, mini_batch_size, lr, nb_epoch])
                            
                            print(f'Final error train comparison: {errors_train[-1,0]:0.2f}')
                            print(f'Final error test comparison: {errors_test[-1,0]:0.2f}')
                            print('\n\n')


    return   torch.tensor(errors_train_comparison), torch.tensor(errors_test_comparison), torch.tensor(errors_train_digits), torch.tensor(errors_test_digits)
        
#TODO Correct the shuffling
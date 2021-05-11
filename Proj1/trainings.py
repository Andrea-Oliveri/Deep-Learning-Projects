from utils import weight_reset, train_model, train_model_aux, plot_loss

def beta_training(model, train_input, train_target, train_classes, test_input, test_target, test_classes,  betas):
    loss_train_final =[]
    loss_test_final  =[]
    errors_train_final=[]
    errors_test_final =[]

    
    for beta in betas:
        model.apply(weight_reset) #Reset the parameters of the model
        print(f'Computing with beta = {beta} ...')
        if model.__class__.__name__ =='ConvNetAux':
            loss_train, loss_test, errors_train, errors_test=train_model_aux(model, train_input, train_target, train_classes, test_input, test_target, test_classes, beta = beta)
        else : 
            raise Exception('The model does not take into account the beta parameter')

        plot_loss(loss_train, loss_test, errors_train, errors_test)

        loss_train_final.append(loss_train[-1])
        loss_test_final.append(loss_test[-1])
        errors_train_final.append(errors_train[-1])
        errors_test_final.append(errors_test[-1])
        
        print(f'Final error train digit: {errors_train[-1,0]:0.2f}')
        if model.__class__.__name__ =='ConvNetAux':print(f'Final error train comparison: {errors_train[-1,1]:0.2f}')
        print(f'Final error test digit: {errors_test[-1,0]:0.2f}')
        if model.__class__.__name__ =='ConvNetAux':print(f'Final error test comparison: {errors_test[-1,1]:0.2f}\n\n')

    
    return loss_train_final, loss_test_final, errors_train_final,  errors_test_final

    

def lr_training(model, train_input, train_target, train_classes, test_input, test_target, test_classes,  lrs):
    loss_train_final =[]
    loss_test_final  =[]
    errors_train_final=[]
    errors_test_final =[]

    
    for lr in lrs:
        model.apply(weight_reset) #Reset the parameters
        print(f'Computing with learning rate = {lr} ...')
        if model.__class__.__name__ =='ConvNetAux':
            loss_train, loss_test, errors_train, errors_test=train_model_aux(model, train_input, train_target, train_classes, test_input, test_target, test_classes, lr=lr)
        else:
            loss_train, loss_test, errors_train, errors_test=train_model(model, train_input, train_target, test_input, test_target, lr=lr)

        plot_loss(loss_train, loss_test, errors_train, errors_test)

        loss_train_final.append(loss_train[-1])
        loss_test_final.append(loss_test[-1])
        errors_train_final.append(errors_train[-1])
        errors_test_final.append(errors_test[-1])
        
        print(f'Final error train digit: {errors_train[-1,0]:0.2f}')
        if model.__class__.__name__ =='ConvNetAux':print(f'Final error train comparison: {errors_train[-1,1]:0.2f}')
        print(f'Final error test digit: {errors_test[-1,0]:0.2f}')
        if model.__class__.__name__ =='ConvNetAux':print(f'Final error test comparison: {errors_test[-1,1]:0.2f}\n\n')

    
    return loss_train_final, loss_test_final, errors_train_final,  errors_test_final

def mini_batch_size_training(model, train_input, train_target, train_classes, test_input, test_target, test_classes,  mini_batch_sizes):
    loss_train_final =[]
    loss_test_final  =[]
    errors_train_final=[]
    errors_test_final =[]

    
    for mini_batch_size in mini_batch_sizes:
        model.apply(weight_reset) #Reset the parameters
        print(f'Computing with mini batch size = {mini_batch_size} ...')
        if model.__class__.__name__ =='ConvNetAux':
            loss_train, loss_test, errors_train, errors_test=train_model_aux(model, train_input, train_target, train_classes, test_input, test_target, test_classes, mini_batch_size=mini_batch_size)
        else:
            loss_train, loss_test, errors_train, errors_test=train_model(model, train_input, train_target, test_input, test_target, mini_batch_size=mini_batch_size)

        plot_loss(loss_train, loss_test, errors_train, errors_test)

        loss_train_final.append(loss_train[-1])
        loss_test_final.append(loss_test[-1])
        errors_train_final.append(errors_train[-1])
        errors_test_final.append(errors_test[-1])
        
        print(f'Final error train digit: {errors_train[-1,0]:0.2f}')
        if model.__class__.__name__ =='ConvNetAux':print(f'Final error train comparison: {errors_train[-1,1]:0.2f}')
        print(f'Final error test digit: {errors_test[-1,0]:0.2f}')
        if model.__class__.__name__ =='ConvNetAux':print(f'Final error test comparison: {errors_test[-1,1]:0.2f}\n\n')

    
    return loss_train_final, loss_test_final, errors_train_final,  errors_test_final

def epochs_training(model, train_input, train_target, train_classes, test_input, test_target, test_classes,  nb_epochs):
    loss_train_final =[]
    loss_test_final  =[]
    errors_train_final=[]
    errors_test_final =[]

    
    for nb_epoch in nb_epochs:
        model.apply(weight_reset) #Reset the parameters
        print(f'Computing with number of epochs = {nb_epoch} ...')
        if model.__class__.__name__ =='ConvNetAux':
            loss_train, loss_test, errors_train, errors_test=train_model_aux(model, train_input, train_target, train_classes, test_input, test_target, test_classes, nb_epochs=nb_epoch)
        else:
            loss_train, loss_test, errors_train, errors_test=train_model(model, train_input, train_target, test_input, test_target, nb_epochs=nb_epoch)

        plot_loss(loss_train, loss_test, errors_train, errors_test)

        loss_train_final.append(loss_train[-1])
        loss_test_final.append(loss_test[-1])
        errors_train_final.append(errors_train[-1])
        errors_test_final.append(errors_test[-1])
        
        print(f'Final error train digit: {errors_train[-1,0]:0.2f}')
        if model.__class__.__name__ =='ConvNetAux':print(f'Final error train comparison: {errors_train[-1,1]:0.2f}')
        print(f'Final error test digit: {errors_test[-1,0]:0.2f}')
        if model.__class__.__name__ =='ConvNetAux':print(f'Final error test comparison: {errors_test[-1,1]:0.2f}\n\n')

    
    return loss_train_final, loss_test_final, errors_train_final,  errors_test_final

def all_training(model, train_input, train_target, train_classes, test_input, test_target, test_classes,\
                 betas, mini_batch_sizes, lrs, nb_epochs):

    loss_train_final =[]
    loss_test_final  =[]
    errors_train_final=[]
    errors_test_final =[]

    
    for mini_batch_size in mini_batch_sizes:
        for lr in lrs:
            for nb_epoch in nb_epochs:
                if model.__class__.__name__ =='ConvNetAux':
                    for beta in betas :
                        model.apply(weight_reset) #Reset the parameters
                        space=''
                        print(f'Computing with: mini batch size = {mini_batch_size}')
                        print(f'{space:>16}learning rate = {lr}')
                        print(f'{space:>16}number of epoch = {nb_epoch}')
                        print(f'{space:>16}beta = {beta}')
                        loss_train, loss_test, errors_train, errors_test=train_model_aux(model, train_input, train_target, train_classes, test_input, test_target, test_classes, beta=beta, lr=lr, mini_batch_size=mini_batch_size, nb_epochs=nb_epoch)

                        plot_loss(loss_train, loss_test, errors_train, errors_test)

                        loss_train_final.append(loss_train[-1])
                        loss_test_final.append(loss_test[-1])
                        errors_train_final.append(errors_train[-1])
                        errors_test_final.append(errors_test[-1])
                        
                        print(f'Final error train digit: {errors_train[-1,0]:0.2f}')
                        if model.__class__.__name__ =='ConvNetAux':print(f'Final error train comparison: {errors_train[-1,1]:0.2f}')
                        print(f'Final error test digit: {errors_test[-1,0]:0.2f}')
                        if model.__class__.__name__ =='ConvNetAux':print(f'Final error test comparison: {errors_test[-1,1]:0.2f}\n\n')

                else:
                    model.apply(weight_reset) #Reset the parameters
                    space=''
                    print(f'Computing with: mini batch size = {mini_batch_size}')
                    print(f'{space:>16}learning rate = {lr}')
                    print(f'{space:>16}number of epoch = {nb_epoch}')

                    loss_train, loss_test, errors_train, errors_test=train_model(model, train_input, train_target, test_input, test_target,lr=lr, mini_batch_size=mini_batch_size, nb_epochs=nb_epoch)

                    plot_loss(loss_train, loss_test, errors_train, errors_test)

                    loss_train_final.append(loss_train[-1])
                    loss_test_final.append(loss_test[-1])
                    errors_train_final.append(errors_train[-1])
                    errors_test_final.append(errors_test[-1])
                    
                    print(f'Final error train digit: {errors_train[-1,0]:0.2f}')
                    if model.__class__.__name__ =='ConvNetAux':print(f'Final error train comparison: {errors_train[-1,1]:0.2f}')
                    print(f'Final error test digit: {errors_test[-1,0]:0.2f}')
                    if model.__class__.__name__ =='ConvNetAux':print(f'Final error test comparison: {errors_test[-1,1]:0.2f}\n\n')

    
    return loss_train_final, loss_test_final, errors_train_final,  errors_test_final
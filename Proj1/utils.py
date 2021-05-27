import torch

def count_nb_parameters(model):
    '''
    Args: 
        model::[torch.nn.Module]
            Initialized instance of Module
    Return: 
        nb_parameters::[int]
            Returns the total number of parameters of the model
    '''
    nb_parameters = sum([param.numel() for param in model.parameters()])

    return nb_parameters    


def print_results_summary(results, model_name):
    '''
    Computes the mean, std, min and max accuracy of multiples run.
    Args:
        results::[list]
            List containing dictionaries with keys 'train_loss', 'test_loss',
            'train_accuracy', 'test_accuracy' and 'final_weights_epoch' for 
            each run.
        model_name::[str]
            The name of the model which has been trained.
    '''
    # Extract the test_accuracy corresponding to the model which EarlyStopping
    # restored the weights of for each run.
    final_test_accuracies = [res['test_accuracy'][res['final_weights_epoch']] for res in results]
    
    mean_final_weights_epoch = sum([res['final_weights_epoch'] for res in results]) / len(results)
    
    # Comparison accuracy.
    comparison_accuracy = [elem[0] for elem in final_test_accuracies]
    mean_comparison = torch.tensor(comparison_accuracy).mean()
    std_comparison  = torch.tensor(comparison_accuracy).std()
    min_comparison  = torch.tensor(comparison_accuracy).min()
    max_comparison  = torch.tensor(comparison_accuracy).max()
    
    print(f"Results for {model_name}:")
    print(f"    Training stopped improving after {mean_final_weights_epoch} epochs on average")
    print(f"    Mean comparison accuracy: {mean_comparison}")
    print(f"    Std  comparison accuracy: {std_comparison}")
    print(f"    Min  comparison accuracy: {min_comparison}")
    print(f"    Max  comparison accuracy: {max_comparison}")
           
    # Digits accuracy for models which do train with auxiliary loss.
    if "no auxiliary loss" not in model_name:
        digits_accuracy = [[elem[-1] for elem in final_test_accuracies]]
        mean_digit = torch.tensor(digits_accuracy).mean()
        std_digit  = torch.tensor(digits_accuracy).std()
        min_digit  = torch.tensor(digits_accuracy).min()
        max_digit  = torch.tensor(digits_accuracy).max()
        
        print(f"    Mean digits accuracy: {mean_digit}")
        print(f"    Std  digits accuracy: {std_digit}")
        print(f"    Min  digits accuracy: {min_digit}")
        print(f"    Max  digits accuracy: {max_digit}")
    
    print('\n')
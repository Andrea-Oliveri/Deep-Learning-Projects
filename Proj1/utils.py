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


def results_summary(results, use_auxiliary_loss):
    '''
    Computes the mean, std, min and max accuracy of multiples run.
    Args:
        results::[list]
            List containing the dictionaries {'train_loss':,'test_loss':,'train_accuracy':,'test_accuracy':} for each run.
        use_auxiliary_loss::[bool]
            Boolean flag determining if the model uses auxiliary loss.
    Returns:
        mean_comparison::[torch.Tensor]
            Mean accuracy of comparison over all runs.
        std_comparison::[torch.Tensor]
            Standard deviation accuracy of comparison over all runs.
        min_comparison::[torch.Tensor]
            Minimum accuracy of comparison over all runs.
        max_comparison::[torch.Tensor]
            Maximum accuracy of comparison over all runs.
        mean_digit::[torch.Tensor]
            Mean accuracy of digit over all runs.
        std_digit::[torch.Tensor]
            Standard deviation accuracy of digit over all runs.
        min_digit::[torch.Tensor]
            Minimum accuracy of digit over all runs.
        max_digit::[torch.Tensor]
            Maximum accuracy of digit over all runs.       
    '''
    final_test_accuracies = [res['test_accuracy'][res['final_weights_epoch']] for res in results]
    
    # Comparison accuracy.
    comparison_accuracy = [elem[0] for elem in final_test_accuracies]
    mean_comparison = torch.tensor(comparison_accuracy).mean()
    std_comparison  = torch.tensor(comparison_accuracy).std()
    min_comparison  = torch.tensor(comparison_accuracy).min()
    max_comparison  = torch.tensor(comparison_accuracy).max()
        
    # Digits accuracy
    if use_auxiliary_loss:
        digits_accuracy = [[elem[-1] for elem in final_test_accuracies]]
        mean_digit = torch.tensor(digits_accuracy).mean()
        std_digit  = torch.tensor(digits_accuracy).std()
        min_digit  = torch.tensor(digits_accuracy).min()
        max_digit  = torch.tensor(digits_accuracy).max() 

        return mean_comparison, std_comparison, min_comparison, max_comparison, mean_digit, std_digit, min_digit, max_digit

    return mean_comparison, std_comparison, min_comparison, max_comparison

    
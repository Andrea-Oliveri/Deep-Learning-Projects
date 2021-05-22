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
    metrics = results[0].keys()
    grouped_metrics = {metric: [res[metric] for res in results] for metric in metrics}

    comparison=[]
    digit=[]

    #Comparison accuracy
    for i in range(len(grouped_metrics['test_accuracy'])):
        comparison.append(grouped_metrics['test_accuracy'][i][-1,0])
    mean_comparison=torch.tensor(comparison).mean()
    std_comparison =torch.tensor(comparison).std()
    min_comparison =torch.tensor(comparison).min()
    max_comparison =torch.tensor(comparison).max()
    #Digit accuracy
    if use_auxiliary_loss:
        for i in range(len(grouped_metrics['test_accuracy'])): 
            digit.append(grouped_metrics['test_accuracy'][i][-1,1])
        mean_digit=torch.tensor(digit).mean()
        std_digit =torch.tensor(digit).std()
        min_digit =torch.tensor(digit).min()
        max_digit =torch.tensor(digit).max() 

        return mean_comparison, std_comparison, min_comparison, max_comparison, mean_digit, std_digit, min_digit, max_digit

    return mean_comparison, std_comparison, min_comparison, max_comparison

    
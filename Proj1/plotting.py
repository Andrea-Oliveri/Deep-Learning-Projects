import torch
import matplotlib.pyplot as plt

def plot_results(results, use_auxiliary_loss):
    '''
    Plots the train/test loss and accuracy for comparison and digit targets of multiple runs

    Args: 
        results::[dict]
            Dictionnary containing 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy' of multiple runs.
        use_auxiliary_loss::[bool]
            Boolean flag determining if model with auxiliary loss is used.

    Returns:
            fig_comparison::[matplotlib.figure.Figure]
                Figure containing the train/test loss/accuracy of the comparison each run
            fig_digit::[matplotlib.figure.Figure]
                Figure containing the train/test loss/accuracy of the digit for each run
    '''     
    metrics = results[0].keys()
    grouped_metrics = {metric: [res[metric] for res in results] for metric in metrics}

    # Draw metrics for comparison for each run individually.
    fig_comparison, axes = plt.subplots(2, 2, figsize = (10, 5))
        
    for ax, metric in zip(axes.ravel(), metrics):
        metric_name = metric.replace('_', ' ').title()+' Comparison'
        ax.set_xlabel("Epoch Number")
        ax.set_ylabel(metric_name)
        
        for metric_training in grouped_metrics[metric]:
            ax.plot(metric_training[:,0])

    fig_comparison.tight_layout()

    #Draws metrics for digit for each run individually
    if use_auxiliary_loss:
        fig_digit, axes = plt.subplots(2, 2, figsize = (10, 5))
            
        for ax, metric in zip(axes.ravel(), metrics):
            metric_name = metric.replace('_', ' ').title()+' Digit'
            ax.set_xlabel("Epoch Number")
            ax.set_ylabel(metric_name)
            
            for metric_training in grouped_metrics[metric]:
                ax.plot(metric_training[:,1])

        fig_digit.tight_layout()

        return fig_comparison, fig_digit

    return fig_comparison
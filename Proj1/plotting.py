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

    '''
    metrics = results[0].keys()
    grouped_metrics = {metric: [res[metric] for res in results] for metric in metrics}

    early_stop_epoch = [torch.argmin(test_loss) for test_loss in grouped_metrics['test_loss']]
    mean_early_stop_epoch = sum(early_stop_epoch) / len(early_stop_epoch)

    # Draw metrics for comparison for each run individually.
    fig, axes = plt.subplots(2, 2, figsize = (10, 5))
        
    for ax, metric in zip(axes.ravel(), metrics):
        metric_name = metric.replace('_', ' ').title()+'Comparison'
        ax.set_xlabel("Epoch Number")
        ax.set_ylabel(metric_name)
        
        for metric_training in grouped_metrics[metric]:
            ax.plot(metric_training[:,0])

        #ax.vlines(mean_early_stop_epoch, *ax.get_ylim(), linestyle = '--', color = 'red')

    plt.tight_layout()
    plt.show()

    #Draws metrics for digit for each run individually
    if use_auxiliary_loss:
        fig, axes = plt.subplots(2, 2, figsize = (10, 5))
            
        for ax, metric in zip(axes.ravel(), metrics):
            metric_name = metric.replace('_', ' ').title()+'Digit'
            ax.set_xlabel("Epoch Number")
            ax.set_ylabel(metric_name)
            
            for metric_training in grouped_metrics[metric]:
                ax.plot(metric_training[:,1])

            #ax.vlines(mean_early_stop_epoch, *ax.get_ylim(), linestyle = '--', color = 'red')

        plt.tight_layout()
        plt.show()
    # # Draw average metrics over all runs.
    # fig, axes = plt.subplots(2, 2, figsize = (10, 5))
        
    # for ax, metric in zip(axes.ravel(), metrics):
    #     metric_name = metric.replace('_', ' ').title()+'Mean'
    #     ax.set_xlabel("Epoch Number")
    #     ax.set_ylabel(metric_name)
        
    #     min_nb_epochs = min([len(m) for m in grouped_metrics[metric]])
    #     metric_same_nb_epochs = [m[:min_nb_epochs] for m in grouped_metrics[metric]]
    #     mean_metric = torch.stack(metric_same_nb_epochs).mean(axis = 0)
    #     ax.plot(mean_metric)
    #     #ax.vlines(mean_early_stop_epoch, *ax.get_ylim(), linestyle = '--', color = 'red')

    
    # plt.tight_layout()
    # plt.show()


    # Collect Statistics
    #metrics_at_early_stop = early_stop_epoch
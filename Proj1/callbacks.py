# -*- coding: utf-8 -*-

import copy


class EarlyStopping(object):
    """Class implementing the Early Stopping callback, which keeps track of
    the best val_loss recorded during training and interrupts training if 
    val_loss didn't improve sufficiently for the last few epochs. Optionally,
    it restores the parameters of the model to what they were at the epoch with
    smallest val_loss. 
    """
    
    def __init__(self, min_improvement = 0., patience = 10, verbose = True,
                 restore_best_weights = True):
        """Constructor of EarlyStopping. Stores desired parameters needed by
        the callback for later use.
        
        Args:
            min_improvement::[float]
                The minimum amount val_loss has to improve in order for
                EarlyStopping to consider it a real improvement and reset
                counter which will eventually stop training.
            patience::[int]
                The number of epochs during which no improvement is observed
                which must be waited before a request to interrupt the
                training.
            verbose::[bool]
                Parameter describing if the callback should print something
                on the terminal when training gets interrupted.
            restore_best_weights::[bool]
                Whether the model weights corresponding to the smalles val_loss
                should be restored when training is interrupted.
        """
        
        self.min_improvement = min_improvement
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_val_loss = float("inf")
        self.best_model_params = None
        self.patience_counter = 0

    def __call__(self, model, val_loss):
        """Special functions to be called once per epoch. Updates the internal
        state of EarlyStopping depending on whether an improvement was observed in
        val_loss and returns True if the training should be interrupted, False
        otherwise. 
        
        Args:
            model::[torch.nn.Module]
                The model currently being trained.
            val_loss::[torch.Tensor]
                The validation loss measured for the current epoch.
                
        Returns:
            stop_training::[bool]
                Boolean describing whether training should be stopped (True) or
                not stopped (False).
            
        """
        if val_loss < self.best_val_loss - self.min_improvement:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            
            if self.restore_best_weights:
                self.best_model_params = copy.deepcopy(model.state_dict())


        else:
            self.patience_counter += 1

            if self.patience_counter >= self.patience:
                if self.verbose:
                    print(
                        "Interrupting Training because no improvement larger "
                        + f"than {self.min_improvement} was observed in the last "
                        + f"{self.patience} epochs: recorded val_loss = "
                        + "{:.4g}, current val_loss = {:.4g}".format(
                            self.best_val_loss, val_loss
                        )
                    )

                if self.restore_best_weights:
                    model.load_state_dict(self.best_model_params)

                    if self.verbose:
                        print("Restoring best model weights.")

                return True

        return False


# -*- coding: utf-8 -*-

import math
from torch import empty


class EarlyStopping(object):
    
    def __init__(self, min_improvement = 0., patience = 10, verbose = True,
                 restore_best_weights = True):
        
        self.min_improvement      = min_improvement
        self.patience             = patience
        self.restore_best_weights = restore_best_weights
        self.verbose              = verbose
        
        self.best_val_loss    = float('inf')
        self.best_model_state = None
        self.patience_counter = 0

        
    def __call__(self, model, val_loss):
        
        if val_loss < self.best_val_loss - self.min_improvement:
            self.best_val_loss    = val_loss
            self.best_model_state = model.state_dict()
            self.patience_counter = 0            
            
        else:
            self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                if self.verbose:
                    print( "Interrupting Training because no improvement was " + \
                          f"observed in the last {self.patience} epochs. Best " + \
                          f"val_loss = {self.best_val_loss}, current val_loss = " + \
                          f"{val_loss}")
                    
                if self.restore_best_weights:
                    model.load_state_dict( self.stored_model_state )
                
                    if self.verbose:
                        print( "Restoring best model weights.")
                
                return True
            
        return False
                
        
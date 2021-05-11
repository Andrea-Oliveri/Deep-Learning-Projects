# -*- coding: utf-8 -*-

import math
from torch import empty


class EarlyStopping(object):
    def __init__(
        self, min_improvement=0.0, patience=10, verbose=True, restore_best_weights=True
    ):

        self.min_improvement = min_improvement
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_val_loss = float("inf")
        self.best_model_params = None
        self.patience_counter = 0

    def __call__(self, model, val_loss):

        if val_loss < self.best_val_loss - self.min_improvement:
            self.best_val_loss = val_loss
            self.best_model_params = model.state_dict()
            self.patience_counter = 0

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

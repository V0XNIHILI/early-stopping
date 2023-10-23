from typing import Optional, Callable

import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0, new_best_callback: Optional[Callable] = None, trace_func=print):
        """
        Early stops the training if validation loss doesn't improve after a given patience.

        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score

            if self.new_best_callback is not None:
                self.new_best_callback()

            return False

        if score < self.best_score + self.delta:
            self.counter += 1

            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                return True
            
            return False

        self.best_score = score
        self.counter = 0

        if self.new_best_callback is not None:
            self.new_best_callback()

        return False


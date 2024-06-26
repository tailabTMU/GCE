import numpy as np
import torch


def expected_calibration_error(y_true, y_pred, n_bins=15, strategy='uniform'):
    """
    Calculates the Expected Calibration Error (ECE) of a classification model.

    Args:
      y_true: True class labels as a 1D array.
      y_pred: Predicted class probabilities as a 1D or 2D array of shape (num_samples, ) for 1D arrays and (num_samples, num_classes) for 2D arrays.
      n_bins: Number of bins to use for binning confidence scores (default: 15).
      strategy: Binning strategy to use. Options are 'uniform' and 'quantile' (default: uniform).

    Returns:
      The ECE value.
    """
    if y_pred.ndim == 1:
        _, num_labels = np.unique(y_true, return_counts=True)
        if num_labels.size != 2:
            raise ValueError(
                "True class labels should be binary."
            )
        predicted_confidences = y_pred
        predicted_labels = (y_pred > 0.5).astype(np.int)
    elif y_pred.ndim == 2:
        predicted_confidences = np.max(y_pred, axis=1)
        predicted_labels = np.argmax(y_pred, axis=1)
    else:
        raise ValueError(
            "Predicted class probabilities should be a 1D or 2D array."
        )
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(predicted_confidences, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )
    bin_ids = np.searchsorted(bins[1:-1], predicted_confidences)
    ece = 0
    for i in range(n_bins):
        props_in_bin = len(predicted_confidences[np.where(bin_ids == i)]) / len(y_pred)
        if props_in_bin > 0:
            avg_bin_confidence = predicted_confidences[np.where(bin_ids == i)].mean()
            bin_accuracy = np.mean(y_true[np.where(bin_ids == i)] == predicted_labels[np.where(bin_ids == i)])
            ece += abs(avg_bin_confidence - bin_accuracy) * props_in_bin

    return ece

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    @Link: https://github.com/Bjarten/early-stopping-pytorch
    @Author: Bjarte Mehus Sunde
    @GitHub: Bjarten
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
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
        self.path = path
        self.trace_func = trace_func
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_epoch = epoch

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def load_checkpoint(self, model):
        """Loads saved model."""
        model.load_state_dict(torch.load(self.path))
        return model

    def get_best_epoch(self):
        return self.best_epoch

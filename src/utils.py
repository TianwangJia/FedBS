import torch
import numpy as np
from scipy.linalg import fractional_matrix_power
import torch.nn as nn

def weights_init(m):
    '''
    Random initialization of the model's parameters
    '''
    # classname = m.__class__.__name__
    # if classname.find("Conv") != -1:
    if isinstance(m,nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
    # elif classname.find('BatchNorm2d') != -1:
    elif isinstance(m,nn.BatchNorm2d):
        # torch.nn.init.normal_(m.weight.data)
        torch.nn.init.normal_(m.weight.data,mean=1.0,std=0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    # elif classname.find('Linear') != -1:
    elif isinstance(m,nn.Linear):
        # torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.xavier_normal_(m.weight.data)

def CalculateOutSize(blocks, channels, samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    x = torch.rand(1, 1, channels, samples)
    for block in blocks:
        block.eval()
        x = block(x)
    shape = x.shape[-2] * x.shape[-1]
    return shape

class Activation(nn.Module):
    '''
    ShallowConcNet Activation Function
    '''
    def __init__(self, type):
        super(Activation, self).__init__()
        self.type = type

    def forward(self, input):
        if self.type == 'square':
            output = input * input
        elif self.type == 'log':
            output = torch.log(torch.clamp(input, min=1e-6)) 
        else:
            raise Exception('Invalid type !')

        return output

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, counter_info=True, is_save=True, early=True):
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
            counter_info(bool): If True, print a message for each counter is increased by 1
                            Default: True
            is_save(bool): If True, save model for each best score improvement
                            Default: True
            early(bool): If True, early stop the training
                            Default: True
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
        
        self.counter_info = counter_info
        self.best_val_acc = 0
        self.best_epoch = 0
        self.is_save = is_save
        self.early = early

    def __call__(self, val_loss, model, val_acc, epoch, global_epochs):

        if self.early:
            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                if self.verbose:
                    self.trace_func(
                        f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
                self.val_loss_min = val_loss
                if self.is_save:
                    self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.counter_info:
                    self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                if self.verbose:
                    self.trace_func(
                        f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
                self.val_loss_min = val_loss
                if self.is_save:
                    self.save_checkpoint(val_loss, model)
                self.counter = 0
        else: 
            if epoch == global_epochs-1:
                self.best_epoch = epoch
                self.best_val_acc = val_acc
                if self.is_save:
                    self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''        
        torch.save(model.state_dict(), self.path)
        

import torch
import torch.nn as nn
from typing import Union
import logging


class earlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, savePath, patience=15, verbose=False, delta=0):
        """
        Args:
            savePath : save path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = savePath
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.bestLoss = None
        self.earlyStop = False
        self.delta = delta

    def __call__(self, valLoss, model):
        if self.bestLoss is None:
            self.saveCheckpoint(valLoss, model)
            self.bestLoss = valLoss
        elif valLoss >= self.bestLoss - self.delta:
            self.counter += 1
            logging.info(f'earlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.earlyStop = True
        else:
            self.saveCheckpoint(valLoss, model)
            self.bestLoss = valLoss
            self.counter = 0

    def saveCheckpoint(self, val_loss, model):
        """ Saves model when validation loss decrease. """
        if self.verbose and self.bestLoss is not None:
            logging.info(f'Validation loss decreased ({self.bestLoss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)


class nseLoss(nn.Module):
    def __init__(self, logLossOpt: bool = False, wLog: Union[float, torch.Tensor] = 0.25, eps=0.01,
                 wStation: Union[list, None] = None):
        """
        :param logLossOpt: whether calculate log nse to account for low flows.
        :param wLog: the weight of log nse if logOpt is True.
        :param eps: a small number to keep numeric stability
        :param wStation: the weight of different hydrological stations. Default value of None means equal weights.
        """
        super().__init__()
        self.logOpt = logLossOpt
        self.wLog = wLog
        self.wStation = wStation
        self.eps = eps

    def nse(self, yTrue: torch.Tensor, yPred: torch.Tensor, spinUp: int = 0):
        """
        :param spinUp: days to spin up model will be excluded when calculating nse.
        :param yTrue: observed streamflow with a shape of [L, N], where L means sequence length and N means the number
                of hydrological stations.
        :param yPred: predicted streamflow with a shape of [L, N].
        :return: nse
        """
        if self.wStation is None:
            wStation = torch.tensor([1]).repeat(yTrue.size(1)).to(yTrue.device)
        else:
            wStation = torch.tensor(self.wStation).to(yTrue.device)

        yTrue, yPred = yTrue[spinUp:], yPred[spinUp:]
        # if the number of nan values exceeds a threshold, the station will be ignored in the later calculation
        idx = torch.tensor(
            [i for i in range(0, yTrue.size(1)) if (torch.isnan(yTrue[:, i]).sum() / yTrue.size(0) < 1 / 5)])
        idx = idx.to(yTrue.device)
        yTrue, yPred = yTrue.index_select(1, idx), yPred.index_select(1, idx)
        wStation = wStation.index_select(0, idx)

        yTrueMean = torch.nanmean(yTrue, dim=0, keepdim=True)
        # pad 0 for nan values in the observation and then mask the 0 values in later calculation
        mask = torch.where(torch.isnan(yTrue), torch.zeros_like(yTrue), torch.ones_like(yTrue))
        yTruePad = torch.where(torch.isnan(yTrue), torch.zeros_like(yTrue), yTrue)
        nseTemp = 1 - ((yPred - yTruePad) ** 2 * mask).sum(dim=0, keepdim=True) / \
                  (((yTruePad - yTrueMean) ** 2 * mask).sum(dim=0, keepdim=True) + self.eps)
        nse = (nseTemp * wStation).sum() / wStation.sum()

        if self.logOpt:
            yTrueLog = torch.log10(torch.sqrt(yTrue) + 0.1)
            yPredLog = torch.log10(torch.sqrt(yPred) + 0.1)
            yTrueLogMean = torch.nanmean(yTrueLog, dim=0, keepdim=True)
            yTrueLogPad = torch.where(torch.isnan(yTrue), torch.zeros_like(yTrue), yTrueLog)
            nseLogTemp = 1 - ((yPredLog - yTrueLogPad) ** 2 * mask).sum(dim=0, keepdim=True) / \
                         (((yTrueLogPad - yTrueLogMean) ** 2 * mask).sum(dim=0, keepdim=True) + self.eps)
            nseLog = (nseLogTemp * wStation).sum() / wStation.sum()

            if isinstance(self.wLog, float):
                self.wLog = torch.tensor(self.wLog)
            return nse * (1 - self.wLog) + nseLog * self.wLog
        else:
            return nse

    def forward(self, yTrue: torch.Tensor, yPred: torch.Tensor, spinUp: int = 0):
        yTrue = yTrue[spinUp:]
        nse = self.nse(yTrue, yPred)
        loss = 1 - nse
        return loss

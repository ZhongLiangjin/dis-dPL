import os
import numpy as np
import torch
import torch.nn as nn
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
import pandas as pd
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
    def __init__(self, logLossOpt: bool = False, wLog: Union[float, torch.Tensor] = 0.25,
                 wStation: Union[list, None] = None):
        """
        :param logLossOpt: whether calculate log nse to account for low flows.
        :param wLog: the weight of log nse if logOpt is True.
        :param wStation: the weight of different hydrological stations. Default value of None means equal weights.
        """
        super().__init__()
        self.logOpt = logLossOpt
        self.wLog = wLog
        self.wStation = wStation

    def nse(self, yTrue: torch.Tensor, yPred: torch.Tensor, spinUp: int):
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
            [i for i in range(0, yTrue.size(1)) if torch.isnan(yTrue[:, i]).sum() / yTrue.size(0) < 1 / 5])
        idx = idx.to(yTrue.device)
        yTrue, yPred = yTrue.index_select(1, idx), yPred.index_select(1, idx)
        wStation = wStation.index_select(0, idx)
        yTrueMean = torch.nanmean(yTrue, dim=0, keepdim=True)
        # pad 0 for nan values in the observation and then mask the 0 values in later calculation
        mask = torch.where(torch.isnan(yTrue), torch.zeros_like(yTrue), torch.ones_like(yTrue))
        yTruePad = torch.where(torch.isnan(yTrue), torch.zeros_like(yTrue), yTrue)
        nseTemp = 1 - ((yPred - yTruePad) ** 2 * mask).sum(dim=0, keepdim=True) / \
                  ((yTruePad - yTrueMean) ** 2 * mask).sum(dim=0, keepdim=True)
        nse = (nseTemp * wStation).sum() / wStation.sum()

        if self.logOpt:
            yTrueLog = torch.log10(torch.sqrt(yTrue) + 0.1)
            yPredLog = torch.log10(torch.sqrt(yPred) + 0.1)
            yTrueLogMean = torch.nanmean(yTrueLog, dim=0, keepdim=True)
            yTrueLogPad = torch.where(torch.isnan(yTrue), torch.zeros_like(yTrue), yTrueLog)
            nseLogTemp = 1 - ((yPredLog - yTrueLogPad) ** 2 * mask).sum(dim=0, keepdim=True) / \
                         ((yTrueLogPad - yTrueLogMean) ** 2 * mask).sum(dim=0, keepdim=True)
            nseLog = (nseLogTemp * wStation).sum() / wStation.sum()

            if isinstance(self.wLog, float):
                self.wLog = torch.tensor(self.wLog)
            return nse * (1 - self.wLog) + nseLog * self.wLog
        else:
            return nse

    def forward(self, yTrue: torch.Tensor, yPred: torch.Tensor, spinUp: int):
        nse = self.nse(yTrue, yPred, spinUp)
        loss = 1 - nse
        return loss


def plotTrnCurve(lossTrnLst: list, lossValLst: list, outPath: str):
    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(range(len(lossTrnLst)), lossTrnLst, color=sns.color_palette('tab10')[0], linewidth=1, label='trn')
    ax.plot(range(len(lossValLst)), lossValLst, color=sns.color_palette('tab10')[1], linewidth=1, label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(outPath)


def saveSimulation(model, loaderVal, loaderTst, outPath, config):
    outDict = defaultdict()
    outVarDict = dict(zip(['Qr', 'Qs', 'Qb', 'Sw', 'E', 'Ssl', 'Sss', 'Paras'],
                          ['Qr', 'Qs', 'Qb', 'Swt', 'E', 'Sslt', 'Ssst', 'Paras']))
    model.eval()
    with torch.no_grad():
        # for validation set
        for x, xn, bsnAttr, rivAttr, y in loaderVal:
            x, xn, y = x.squeeze(0), xn.squeeze(0), y.squeeze(0)
            bsnAttr, rivAttr = bsnAttr.squeeze(0), rivAttr.squeeze(0)
            output = model(x, xn, bsnAttr, rivAttr, staIdx=config['data']['spinUp'][1], mode='analyse')
            yPred = output['Qr'][config['train']['outBsnIdx'], :].permute(1, 0).detach().cpu().numpy()
            yTrue = y.detach().cpu().numpy()
            # calculate metrics
            evaluateFn(true=yTrue, pred=yPred, hydroSta=config['data']['hydStations'], mode='val',
                       spinUp=config['data']['spinUp'][1])
            # save the simulated hydrological variables
            for k, v in outVarDict.items():
                if k is not 'Paras':
                    outDict[k] = output[v].detach().cpu().numpy()[:, config['data']['spinUp'][1]:]

        # for test set
        for x, xn, bsnAttr, rivAttr, y in loaderTst:
            x, xn, y = x.squeeze(0), xn.squeeze(0), y.squeeze(0)
            bsnAttr, rivAttr = bsnAttr.squeeze(0), rivAttr.squeeze(0)
            output = model(x, xn, bsnAttr, rivAttr, staIdx=config['data']['spinUp'][2], mode='analyse')
            yPred = output['Qr'][config['train']['outBsnIdx'], :].permute(1, 0).detach().cpu().numpy()
            yTrue = y.detach().cpu().numpy()
            evaluateFn(true=yTrue, pred=yPred, hydroSta=config['data']['hydStations'], mode='tst',
                       spinUp=config['data']['spinUp'][2])
            for k, v in outVarDict.items():
                if k is not 'Paras':
                    outTemp = output[v].detach().cpu().numpy()[:, config['data']['spinUp'][2]:]
                    outDict[k] = np.concatenate((outDict[k], outTemp), axis=1)
                else:
                    for paraName, paraValue in output[v].items():
                        outDict[paraName] = paraValue.detach().cpu().numpy()[:, config['data']['spinUp'][2]:] if \
                            paraName is 'dynamic' else paraValue.detach().cpu().numpy()
    with open(os.path.join(config['out'], outPath), 'wb') as f:
        f.write(pickle.dumps(outDict))


def evaluateFn(true, pred, hydroSta, mode, spinUp):
    logging.info('*' * 100)
    logging.info(f'Calculating metrics on {mode} set')

    for i, station in enumerate(hydroSta):
        trueTemp, predTemp = true[spinUp:, i], pred[spinUp:, i]
        df = pd.DataFrame.from_records([trueTemp, predTemp], index=['true', 'pred']).T
        df.dropna(inplace=True)
        # calculate nse
        numerator = np.sum((df['pred'] - df['true']) ** 2)
        denominator = np.sum((df['true'] - np.mean(df['true'])) ** 2)
        nse = 1 - numerator / denominator
        # calculate pearson's correlation coefficiency
        r = np.corrcoef(df['pred'].values, df['true'].values)[0, 1]
        # calculate PBIAS
        pbias = 100 * (df['pred'] - df['true']).sum() / df['true'].sum()

        logging.info(f'For streamflow at {station} station: nse={nse: .3f}, r={r: .3f}, PBIAS={pbias: .3f}')

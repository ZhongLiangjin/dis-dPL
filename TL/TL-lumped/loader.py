import numpy as np
import pickle
import pandas as pd
from typing import List, Union, Tuple
import torch
from torch.utils.data import dataset, dataloader


class DataLoader:
    def __init__(self, xTrainDataFile: str, xTestDataFile: str, flowFile: str, periods: List[List[str]], spinUp: int,
                 hydStation: Union[str, List[str]], seqLen: int = 365, winSize: int = 365,
                 device: Union[str, torch.device] = 'cpu', excludeBsnAttrVar: Union[None, str, List[str]] = None):
        """
        :param xTestDataFile: path of the data.pkl containing the mean forcing and attributes of 69 sub-basins.
        :param xTrainDataFile: path of the data.pkl containing the mean forcing and attributes of 6 sub-basins based on
            the 6 hydrological stations.
        :param flowFile: path of the streamflow csv file.
        :param periods: a list used to split training, validation, and test periods.
        :param spinUp: a list containing days used to initiate model for training, validation and test sets,
            respectively.
        :param seqLen: sequence length for training set.
        :param winSize: window size to generate sequence.
        :param hydStation: the streamflow at which hydrological stations are selected to train model.
        :param device: whether to use gpu to accelerate.
        :param excludeBsnAttrVar: the excluded variables in basin attributes.
        """
        super().__init__()
        # load input data
        with open(xTrainDataFile, "rb") as f:
            self.xTrainData = pickle.load(f)
        with open(xTestDataFile, "rb") as f:
            self.xTestData = pickle.load(f)
        # read streamflow data
        self.flow = pd.read_csv(flowFile, index_col='DATE', parse_dates=True)
        area = dict(zip(list(self.xTrainData.keys()),
                        [self.xTrainData[bsn]['attrs']['area'] for bsn in self.xTrainData.keys()]))
        for bsn in self.flow.columns:
            self.flow.loc[:, bsn] = self.flow.loc[:, bsn] * 24 * 3600 / (area[bsn] * 1000)  # convert m3/s to mm/d

        # split training, validation, and test sets
        self.trnPer, self.valPer, self.tstPer = periods
        self.dataTrn, self.dataVal, self.dataTst = self.splitSets(hydStation, excludeBsnAttrVar)

        # calculate mean and std, and do normalization
        self.statDict = calMeanStd(forDataTrn=self.dataTrn['forData'], forVarLst=self.dataTrn['forVarLst'],
                                   bsnAttr=self.dataTrn['bsnAttr'], bsnAttrVarLst=self.dataTrn['bsnAttrVarLst'])
        for dataDict in [self.dataTrn, self.dataVal, self.dataTst]:
            dataDict['forDataNorm'] = self.transNorm(dataDict['forData'], dataDict['forVarLst'], toNorm=True)
            dataDict['bsnAttr'] = self.transNorm(dataDict['bsnAttr'], dataDict['bsnAttrVarLst'], toNorm=True)

        # generate sequences
        self.generateSequence(spinUp=spinUp, winSize=winSize, seqLen=seqLen, hydStation=hydStation)

        # get dataset and dataloader
        dsTrn = MyDataset(self.dataTrn, device)
        dsVal = MyDataset(self.dataVal, device)
        dsTst = MyDataset(self.dataTst, device)
        self.loaderTrn = dataloader.DataLoader(dataset=dsTrn, batch_size=1, shuffle=True)
        self.loaderVal = dataloader.DataLoader(dataset=dsVal, batch_size=1, shuffle=False)
        self.loaderTst = dataloader.DataLoader(dataset=dsTst, batch_size=1, shuffle=False)

    def splitSets(self, hydStation: Union[str, List[str]], excludeBsnAttrVar: Union[None, str, List[str]] = None)\
            -> Tuple[dict, dict, dict]:
        # get forData data for the 6 sub-basins first
        bsns6 = list(self.xTrainData.keys())
        tRange = list(self.xTrainData[bsns6[0]]['forcing'].index)
        forVarLst = list(self.xTrainData[bsns6[0]]['forcing'].columns)
        forData6 = np.concatenate([np.expand_dims(v['forcing'].values, 0) for k, v in self.xTrainData.items()], axis=0)
        # forData for the 69 sub-basins
        bsns69 = list(self.xTestData.keys())
        forData69 = np.concatenate([np.expand_dims(v['forcing'].values, 0) for k, v in self.xTestData.items()], axis=0)

        # get basin attributes data
        tempBsnAttrVarLst = list(self.xTrainData[bsns6[0]]['attrs'].keys())
        if excludeBsnAttrVar is None:  # No excluded variables
            bsnAttrVarLst = tempBsnAttrVarLst
        else:  # exclude some variables
            if isinstance(excludeBsnAttrVar, str):
                excludeBsnAttrVar = [excludeBsnAttrVar]
            assert isinstance(excludeBsnAttrVar, list)
            bsnAttrVarLst = [var for var in tempBsnAttrVarLst if var not in excludeBsnAttrVar]
        bsnAttr6 = np.zeros((len(bsns6), len(bsnAttrVarLst)))
        for i, bsn in enumerate(bsns6):
            bsnAttr6[i] = np.array([v for k, v in self.xTrainData[bsn]['attrs'].items() if k in bsnAttrVarLst])

        bsnAttr69 = np.zeros((len(bsns69), len(bsnAttrVarLst)))
        for i, bsn in enumerate(bsns69):
            bsnAttr69[i] = np.array([v for k, v in self.xTestData[bsn]['attrs'].items() if k in bsnAttrVarLst])

        # get runoff data
        if isinstance(hydStation, str):
            hydStation = [hydStation]
        yData = self.flow[hydStation].values
        yVarLst = hydStation  # hydrological stations

        # split training, validation, and test sets
        tRangeTrn = pd.date_range(self.trnPer[0], self.trnPer[1])
        forTimeIdxTrn = np.arange((tRangeTrn[0] - tRange[0]).days, (tRangeTrn[-1] - tRange[0]).days + 1)
        bsnIdxTrn = [bsns6.index(bsn) for bsn in hydStation]
        forDataTrn = forData6[bsnIdxTrn][:, forTimeIdxTrn]
        yDataTrn = self.flow[hydStation][self.flow.index.isin(tRangeTrn)].values
        bsnAttrTrn = bsnAttr6[bsnIdxTrn]
        dataTrn = {'bsns': bsns6, 'tRange': tRangeTrn, 'forVarLst': forVarLst, 'forData': forDataTrn,
                   'bsnAttrVarLst': bsnAttrVarLst, 'bsnAttr': bsnAttrTrn,
                   'yData': yDataTrn, 'yVarLst': yVarLst}

        tRangeVal = pd.date_range(self.valPer[0], self.valPer[1])
        forTimeIdxVal = np.arange((tRangeVal[0] - tRange[0]).days,
                                  (tRangeVal[-1] - tRange[0]).days + 1)  # index to get forcing
        forDataVal6 = forData6[:, forTimeIdxVal]
        forDataVal69 = forData69[:, forTimeIdxVal]
        forDataVal = np.concatenate((forDataVal6, forDataVal69), axis=0)
        yDataVal = self.flow[hydStation][self.flow.index.isin(tRangeVal)].values
        dataVal = {'bsns': bsns6, 'tRange': tRangeVal, 'forVarLst': forVarLst, 'forData': forDataVal,
                   'bsnAttrVarLst': bsnAttrVarLst, 'bsnAttr': bsnAttrTrn, 'yData': yDataVal, 'yVarLst': yVarLst}

        tRangeTst = pd.date_range(self.tstPer[0], self.tstPer[1])
        forTimeIdxTst = np.arange((tRangeTst[0] - tRange[0]).days, (tRangeTst[-1] - tRange[0]).days + 1)
        forDataTst6 = forData6[:, forTimeIdxTst]
        forDataTst69 = forData69[:, forTimeIdxTst]
        forDataTst = np.concatenate((forDataTst6, forDataTst69), axis=0)
        bsnAttrTst = np.concatenate((bsnAttr6, bsnAttr69), axis=0)
        yDataTst = self.flow[hydStation][self.flow.index.isin(tRangeTst)].values
        dataTst = {'bsns': bsns6, 'tRange': tRangeTst, 'forVarLst': forVarLst, 'forData': forDataTst,
                   'bsnAttrVarLst': bsnAttrVarLst, 'bsnAttr': bsnAttrTst, 'yData': yDataTst, 'yVarLst': yVarLst}

        return dataTrn, dataVal, dataTst

    def transNorm(self, x: np.ndarray, varLst: Union[str, List[str]], toNorm: bool = True):
        """
        :param x: forcing data, basin attributes or river attributes.
        :param varLst: list or string of variable names.
        :param toNorm: whether to do normalization or reverse normalization.
        """
        if type(varLst) is str:
            varLst = [varLst]
        out = np.zeros(x.shape)
        for k in range(len(varLst)):
            var = varLst[k]
            stat = self.statDict[var]
            if toNorm is True:  # do normalization
                if len(x.shape) == 3:
                    if var in ['pr', 'flow']:
                        temp = np.log10(np.sqrt(x[:, :, k]) + 0.1)
                        out[:, :, k] = (temp - stat[0]) / stat[1]
                    else:
                        out[:, :, k] = (x[:, :, k] - stat[0]) / stat[1]
                elif len(x.shape) == 2:
                    if var in ['pr', 'flow']:
                        temp = np.log10(np.sqrt(x[:, k]) + 0.1)
                        out[:, k] = (temp - stat[0]) / stat[1]
                    else:
                        out[:, k] = (x[:, k] - stat[0]) / stat[1]
            else:  # reverse normalization
                if len(x.shape) == 3:
                    out[:, :, k] = x[:, :, k] * stat[1] + stat[0]
                    if var in ['pr', 'flow']:
                        tempTrans = np.power(10, out[:, :, k]) - 0.1
                        tempTrans[tempTrans < 0] = 0  # set negative as zero
                        out[:, :, k] = tempTrans ** 2
                elif len(x.shape) == 2:
                    out[:, k] = x[:, k] * stat[1] + stat[0]
                    if var in ['pr', 'flow']:
                        tempTrans = np.power(10, out[:, k]) - 0.1
                        tempTrans[tempTrans < 0] = 0
                        out[:, k] = tempTrans ** 2
        return out

    def generateSequence(self, spinUp: Union[int, List[int]], winSize: int, seqLen: int, hydStation: Union[str, List[str]]):
        if isinstance(spinUp, int):
            spinUp = [spinUp] * 3
        shapeXnTrn, shapeXTrn = self.dataTrn['forDataNorm'].shape, self.dataTrn['forData'].shape
        numSeq = int((shapeXTrn[1] - seqLen) / winSize) + 1
        xTrn = np.zeros((numSeq, shapeXTrn[0], seqLen + spinUp[0], shapeXTrn[2]))  # forcing to hydrological model
        xnTrn = np.zeros((numSeq, shapeXnTrn[0], seqLen + spinUp[0], shapeXnTrn[2]))  # normalized forcing
        yTrn = np.zeros((numSeq, seqLen + spinUp[0], self.dataTrn['yData'].shape[1]))  # streamflow
        idxStart = 0
        for i in range(numSeq):
            if idxStart < spinUp[0]:
                if idxStart == 0:
                    forDataSpin = self.dataTrn['forData'][:, idxStart - spinUp[0]:]
                    forNormDataSpin = self.dataTrn['forDataNorm'][:, idxStart - spinUp[0]:]
                    yDataSpin = self.dataTrn['yData'][idxStart - spinUp[0]:]
                else:
                    forDataSpin = np.concatenate((self.dataTrn['forData'][:, idxStart - spinUp[0]:],
                                                  self.dataTrn['forData'][:, :idxStart]), axis=1)
                    forNormDataSpin = np.concatenate((self.dataTrn['forDataNorm'][:, idxStart - spinUp[0]:],
                                                      self.dataTrn['forDataNorm'][:, :idxStart]), axis=1)
                    yDataSpin = np.concatenate((self.dataTrn['yData'][idxStart - spinUp[0]:],
                                                self.dataTrn['yData'][:idxStart]), axis=0)
            else:
                forDataSpin = self.dataTrn['forData'][:, idxStart - spinUp[0]:idxStart]
                forNormDataSpin = self.dataTrn['forDataNorm'][:, idxStart - spinUp[0]:idxStart]
                yDataSpin = self.dataTrn['yData'][idxStart - spinUp[0]:idxStart]

            xTrn[i] = np.concatenate((forDataSpin, self.dataTrn['forData'][:, idxStart: idxStart + seqLen]), axis=1)
            xnTrn[i] = np.concatenate((forNormDataSpin, self.dataTrn['forDataNorm'][:, idxStart: idxStart + seqLen]),
                                      axis=1)
            yTrn[i] = np.concatenate((yDataSpin, self.dataTrn['yData'][idxStart: idxStart + seqLen]), axis=0)
            idxStart += winSize

        # validation set
        xValSpin = self.dataTrn['forData'][:, -spinUp[1]:]
        xnValSpin = self.dataTrn['forDataNorm'][:, -spinUp[1]:]
        yValSpin = self.dataTrn['yData'][-spinUp[1]:]

        # test set
        xTstSpin = self.dataVal['forData'][:, -spinUp[2]:]
        xnTstSpin = self.dataVal['forDataNorm'][:, -spinUp[2]:]
        yTstSpin = self.dataVal['yData'][-spinUp[2]:]

        # update the training, validation and test sets
        self.dataTrn['forData'] = xTrn
        self.dataTrn['forDataNorm'] = xnTrn
        self.dataTrn['yData'] = yTrn
        self.dataTrn['spinUpLen'] = spinUp[0]

        bsnIdx = [self.dataVal['bsns'].index(station) for station in hydStation]
        self.dataVal['forData'] = self.dataVal['forData'][bsnIdx]
        self.dataVal['forData'] = np.expand_dims(np.concatenate((xValSpin, self.dataVal['forData']), axis=1), axis=0)
        self.dataVal['forDataNorm'] = self.dataVal['forDataNorm'][bsnIdx]
        self.dataVal['forDataNorm'] = np.expand_dims(np.concatenate((xnValSpin, self.dataVal['forDataNorm']), axis=1),
                                                     axis=0)
        self.dataVal['yData'] = np.expand_dims(np.concatenate((yValSpin, self.dataVal['yData']), axis=0), axis=0)
        self.dataVal['spinUpLen'] = spinUp[1]

        self.dataTst['forData'] = np.expand_dims(np.concatenate((xTstSpin, self.dataTst['forData']), axis=1), axis=0)
        self.dataTst['forDataNorm'] = np.expand_dims(np.concatenate((xnTstSpin, self.dataTst['forDataNorm']), axis=1),
                                                     axis=0)
        self.dataTst['yData'] = np.expand_dims(np.concatenate((yTstSpin, self.dataTst['yData']), axis=0), axis=0)
        self.dataTst['spinUpLen'] = spinUp[2]


class MyDataset(dataset.Dataset):
    def __init__(self, dataDict: dict, device: Union[str, torch.device]):
        self.x = torch.tensor(dataDict['forData'], device=device, dtype=torch.float32).permute(0, 2, 1, 3)
        self.xn = torch.tensor(dataDict['forDataNorm'], device=device, dtype=torch.float32).permute(0, 2, 1, 3)
        self.bsnAttr = torch.tensor(dataDict['bsnAttr'], device=device, dtype=torch.float32)
        self.y = torch.tensor(dataDict['yData'], device=device, dtype=torch.float32)

    def __getitem__(self, index):
        xTrain = self.x[index]
        xn = self.xn[index]
        bsnAttr = self.bsnAttr
        zTrain = torch.concat((xn, bsnAttr.unsqueeze(0).expand(len(xn), -1, -1)), dim=-1)
        y = self.y[index]
        return xTrain, zTrain, y

    def __len__(self):
        return len(self.x)


def calMeanStd(forDataTrn: np.ndarray, forVarLst: List[str], bsnAttr: np.ndarray, bsnAttrVarLst: List[str]) -> dict:
    """
    :param forDataTrn: forcing data on the training set with a shape of [N, L, F].
    :param forVarLst: a list of forcing variable names.
    :param bsnAttr: basin attributes data with a shape of [N, F].
    :param bsnAttrVarLst: a list of basin attributes variable names.

    :return: a dictionary storing the mean and std for forcing, basin attributes, and river attributes.
    """
    statDict = {}
    # forcing data
    for k, var in enumerate(forVarLst):
        if var in ['pr', 'flow']:
            statDict[var] = calStatGamma(forDataTrn[:, :, k])
        else:
            statDict[var] = calStat(forDataTrn[:, :, k])
    # basin attributes
    for k, var in enumerate(bsnAttrVarLst):
        statDict[var] = calStat(bsnAttr[:, k])

    return statDict


def calStatGamma(x):  # for daily streamflow and precipitation
    a = x.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    b = np.log10(np.sqrt(b) + 0.1)  # do some tranformation to change gamma characteristics
    # p10 = np.percentile(b, 10).astype(float)
    # p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(np.float32)
    std = np.std(b).astype(np.float32)
    if std < 0.001:
        std = 1
    return [mean, std]


def calStat(x):
    a = x.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    # p10 = np.percentile(b, 10).astype(float)
    # p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(np.float32)
    std = np.std(b).astype(np.float32)
    if std < 0.001:
        std = 1
    return [mean, std]


if __name__ == '__main__':
    loader = DataLoader(xTestDataFile='./data/xdata_69basins.pkl', xTrainDataFile='./data/xdata_6basins.pkl',
                        flowFile='./data/streamflow.csv',
                        periods=[['1960-1-1', '1989-12-31'], ['1990-1-1', '1999-12-31'], ['2000-1-1', '2019-12-31']],
                        spinUp=730, seqLen=2190, winSize=2190, hydStation=['TNH', 'MAQ', 'JIM', 'HHY'])

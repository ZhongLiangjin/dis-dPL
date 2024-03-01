from collections import defaultdict
from typing import List, Union
import numpy as np
import pandas as pd
import pickle
import json
from matplotlib.ticker import ScalarFormatter
import os
import geopandas as gpd


class Evaluator:
    def __init__(self, ensembleFolder, flowFile, valDataFile, basinFile, modelTyp):
        """
        :param ensembleFolder: directory of ensemble members.
        :param flowFile: observed runoff csv file.
        :param basinFile: sub-basin shape file.
        :param valDataFile: a pickle file storing remote sensing snow depth and GLEAM ET.
        :param modelTyp: tDPL model or dDPL model
        """
        assert modelTyp in ['tDPL', 'dDPL', 'dPL']
        self.modelTyp = modelTyp

        ensembleDict = defaultdict(dict)
        for file in os.listdir(ensembleFolder):
            if file.startswith('seed'):
                seed = file.split('_')[1]
                with open(os.path.join(ensembleFolder, file, 'config.json'), 'r') as f:
                    self.config = json.load(f)
                # read simulation
                with open(os.path.join(ensembleFolder, file, 'simulation.pkl'), 'rb') as f:
                    simulation = pickle.load(f)

                if self.modelTyp == 'dDPL':
                    tRange = pd.date_range(self.config['data']['periods'][1][0], self.config['data']['periods'][2][1])
                    for k, v in simulation.items():
                        if v.shape == (69, 10957):  # load Q and ET
                            ensembleDict[seed][k] = pd.DataFrame(v.T, index=tRange, columns=range(0, 69))
                else:
                    tRange = pd.date_range(self.config['data']['periods'][2][0], self.config['data']['periods'][2][1])
                    for k, v in simulation.items():
                        ensembleDict[seed][k] = pd.DataFrame(v, index=tRange, columns=range(0, v.shape[1]))

        seeds = list(ensembleDict.keys())
        variables = ensembleDict[seeds[0]].keys()
        for var in variables:
            mean = np.concatenate([np.expand_dims(ensembleDict[seed][var].values, 0) for seed in seeds],
                                  axis=0).mean(axis=0)
            ensembleDict['mean'][var] = pd.DataFrame(mean, index=tRange, columns=ensembleDict[seeds[0]][var].columns)
        self.sim = ensembleDict

        # read streamflow data
        flow = pd.read_csv(flowFile, index_col='DATE', parse_dates=True)
        self.flow = flow[flow.index.isin(tRange)]

        # get validated data including et and snow depth
        with open(valDataFile, 'rb') as f:
            self.valData = pickle.load(f)

        # get basic geometric information of 69 sub-basins.
        self.basins = gpd.read_file(basinFile)

        # get metrics for Q, ET, and snow depth
        self.metric = defaultdict(lambda: defaultdict(dict))
        for key in self.sim.keys():
            self.metric['Q'][key] = self.evalRunoff(tRange='tst', mode=key)
            self.metric['ET'][key] = self.evalET(tRange=['2000-1-1', '2019-12-31'], tScale='m', dataTyp='GLDAS',
                                                 mode=key)
            self.metric['Snow depth'][key] = self.evalSnowDepth(tRange=['2000-1-1', '2019-12-31'], tScale='d', mode=key)

    def evalRunoff(self, tRange: Union[List[str], str] = 'tst', mode: str = 'mean'):
        """
        calculate performance for streamflow.
        :param tRange: plot runoff in the given time range, could be 'tst' or ['2000-1-1', '2012-12-31'].
        :param mode: calculate metrics from ensemble mean or single .
        """
        # get date range
        if isinstance(tRange, str):
            assert tRange in ['trn', 'val', 'tst']
            tRange = self.config['data']['periods'][['trn', 'val', 'tst'].index(tRange)]
            period = pd.date_range(tRange[0], tRange[1])
        else:
            period = pd.date_range(tRange[0], tRange[1])

        # select data from TNH, JUG, MAQ, MET, JIM, and HHY stations
        area = {'HHY': 20930, 'JIM': 45019, 'JUG': 98414, 'MAQ': 86048, 'MET': 59655, 'TNH': 121972}
        if self.modelTyp == 'dDPL':
            dfPred = self.sim[mode]['Qr'][[13, 45, 53, 35, 23, 0]]
            dfPred = dfPred.rename(columns={13: 'HHY', 45: 'JIM', 53: 'MET', 35: 'MAQ', 23: 'JUG', 0: 'TNH'})
        else:
            dfPred = self.sim[mode]['Qr']
            dfPred = dfPred.rename(columns={5: 'HHY', 4: 'JIM', 3: 'MET', 2: 'MAQ', 1: 'JUG', 0: 'TNH'})
        dfPred = dfPred[dfPred.index.isin(period)]
        dfTrue = self.flow[self.flow.index.isin(period)]

        metricDict = defaultdict(dict)
        for i, station in enumerate(list(dfPred.columns)[::-1]):
            unit = 24 * 3600 / (area[station] * 1000)
            if station == 'MET':  # only evaluate the performance in month 5~10
                if self.modelTyp == 'dDPL':
                    pred = dfPred[station][(dfPred.index.month >= 5) & (dfPred.index.month <= 10)] * unit
                else:
                    pred = dfPred[station][(dfPred.index.month >= 5) & (dfPred.index.month <= 10)]
                true = dfTrue[station][(dfTrue.index.month >= 5) & (dfTrue.index.month <= 10)] * unit
            else:
                if self.modelTyp == 'dDPL':
                    pred, true = dfPred[station] * unit, dfTrue[station] * unit
                else:
                    pred, true = dfPred[station], dfTrue[station] * unit

            pred.name, true.name = 'pred', 'true'
            dfTemp = pd.concat([pred, true], axis=1)
            dfTemp.dropna(inplace=True)  # drop nan value
            nse, r, pbias, kge, rmse, nseLog = evalFn(true=dfTemp['true'].values, pred=dfTemp['pred'].values)
            dictTemp = {'NSE': nse, 'r': r, 'PBIAS': pbias, 'KGE': kge, 'RMSE': rmse, 'log NSE': nseLog}
            metricDict[station] = dictTemp
        return metricDict

    def evalSnowDepth(self, tRange: List[str], tScale: str = 'm', mode: str = 'mean'):
        """
        :param tRange: time range with the form of [start time, end time]
        :param tScale: time scale, could be 'd' (abbreviated for daily), 'm' (monthly), and 'a' (annual).
        :param mode: calculate metrics from ensemble mean or single member.
        """
        tRange = pd.date_range(tRange[0], tRange[1])
        obsSdTemp = self.valData['snowDepth'][self.valData['snowDepth'].index.isin(tRange)]
        obsSdTemp.dropna(inplace=True)
        predSdTemp = self.sim[mode]['Sw'][self.sim[mode]['Sw'].index.isin(obsSdTemp.index)]

        # get data of targeted time scale
        if tScale == 'd':
            obsSd, predSd = obsSdTemp, predSdTemp
        elif tScale == 'm':
            obsSd = obsSdTemp.groupby([obsSdTemp.index.year, obsSdTemp.index.month]).mean()
            predSd = predSdTemp.groupby([predSdTemp.index.year, predSdTemp.index.month]).mean()
        elif tScale == 'a':
            obsSd = obsSdTemp.groupby(obsSdTemp.index.year).mean()
            predSd = predSdTemp.groupby(predSdTemp.index.year).mean()
        else:
            raise ValueError('tRange must be d, m, or a')

        dfMetric = pd.DataFrame(columns=['r', 'RMSE'], index=range(0, 69))
        for idx in dfMetric.index:
            obs = obsSd.loc[:, f'basin_{idx}'].values.astype('float')
            pred = predSd.loc[:, idx].values.astype('float')
            dfMetric.loc[idx, 'r'] = np.corrcoef(obs, pred)[0, 1]
            dfMetric.loc[idx, 'RMSE'] = np.sqrt(np.mean((obs - pred) ** 2))

        return {'r': dfMetric['r'].values, 'RMSE': dfMetric['RMSE'].values}

    def evalET(self, tRange: List[str], tScale: str = 'm', dataTyp: str = 'GLEAM', mode: str = 'mean'):
        """
        :param tRange: time range with the form of [start time, end time]
        :param tScale: time scale, could be 'd' (abbreviated for daily), 'm' (monthly), and 'a' (annual).
        :param mode: calculate metrics from ensemble mean or single member.
        """
        assert dataTyp in ['GLEAM', 'GLDAS']
        tRange = pd.date_range(tRange[0], tRange[1])
        obsEtTemp = self.valData[f'{dataTyp}Et'][self.valData[f'{dataTyp}Et'].index.isin(tRange)]
        predEtTemp = self.sim[mode]['E'][self.sim[mode]['E'].index.isin(obsEtTemp.index)]

        # get data of targeted time scale
        if dataTyp == 'GLEAM':
            if tScale == 'd':
                obsEt, predEt = obsEtTemp, predEtTemp
            elif tScale == 'm':
                obsEt = obsEtTemp.groupby([obsEtTemp.index.year, obsEtTemp.index.month]).mean()
                predEt = predEtTemp.groupby([predEtTemp.index.year, predEtTemp.index.month]).mean()
            elif tScale == 'a':
                obsEt = obsEtTemp.groupby(obsEtTemp.index.year).mean()
                predEt = predEtTemp.groupby(predEtTemp.index.year).mean()
            else:
                raise ValueError(f'For {dataTyp} ET, the tRange must be d, m, or a')
        else:
            if tScale == 'm':
                obsEt = obsEtTemp.groupby([obsEtTemp.index.year, obsEtTemp.index.month]).mean()
                predEt = predEtTemp.groupby([predEtTemp.index.year, predEtTemp.index.month]).mean()
            elif tScale == 'a':
                obsEt = obsEtTemp.groupby(obsEtTemp.index.year).mean()
                predEt = predEtTemp.groupby(predEtTemp.index.year).mean()
            else:
                raise ValueError(f'For {dataTyp} ET, the tRange must be m, or a')

        dfMetric = pd.DataFrame(columns=['r', 'RMSE'], index=range(0, 69))
        for idx in dfMetric.index:
            obs = obsEt.loc[:, f'basin_{idx}'].values.astype('float')
            pred = predEt.loc[:, idx].values.astype('float')
            dfMetric.loc[idx, 'r'] = np.corrcoef(obs, pred)[0, 1]
            dfMetric.loc[idx, 'RMSE'] = np.sqrt(np.mean((obs - pred) ** 2))

        return {'r': dfMetric['r'].values, 'RMSE': dfMetric['RMSE'].values}

def evalFn(true: np.ndarray, pred: np.ndarray):
    # calculate nse
    numerator = np.sum((pred - true) ** 2)
    denominator = np.sum((true - np.mean(true)) ** 2)
    nse = 1 - numerator / denominator
    # calculate pearson's correlation coefficient
    r = np.corrcoef(pred, true)[0, 1]
    # calculate the percent bias
    pBias = 100 * (pred - true).sum() / true.sum()
    # calculate KGE
    beta = np.mean(pred) / np.mean(true)
    gamma = (np.std(pred) / np.mean(pred)) / (np.std(true) / np.mean(true))
    kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    # calculate RMSE
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    # calculate log nse
    predLog, trueLog = np.log10(pred + 0.1), np.log10(true + 0.1)
    numerator = np.sum((predLog - trueLog) ** 2)
    denominator = np.sum((trueLog - np.mean(trueLog)) ** 2)
    nseLog = 1 - numerator / denominator
    return nse, r, pBias, kge, rmse, nseLog



class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here


if __name__ == "__main__":
    rootDir = '../checkpoints/Types'
    meanOpt = True  # determine whether plot mean performance for ET and snow depth
    modelNames = ['TL-a', 'TL-b', 'TL-c', 'TL-d']
    stations = ['TNH', 'JUG', 'MAQ', 'MET', 'JIM']
    #  initiate a dict to store NSE, r, and PBIAS of streamflow, r and RMSE of ET and snow depth for five models
    modelMetric = defaultdict(dict)
    for model in modelNames:
        print(f'Calculating metrics for model {model}')
        modelDir = os.path.join(rootDir, model)
        evaluator = Evaluator(ensembleFolder=modelDir,
                              basinFile='../data/sub-basins/watershed.shp',
                              flowFile='../data/streamflow.csv',
                              valDataFile='../data/valData.pkl',
                              modelTyp='dDPL')
        modelMetric[model] = evaluator.metric
    seeds = [k for k in modelMetric[modelNames[0]]['Q'].keys() if k is not 'mean']
import time
import torch
import torch.backends.cudnn as cudnn
from utils.trainTools import earlyStopping, nseLoss, plotTrnCurve, saveSimulation
from pathlib import Path
import os
import json
import argparse
from typing import Union
from collections import defaultdict
from data.loader import DataLoader
from model.dDPLExpHYDRO import Net
import random
import numpy as np
import logging


def trainModel(model, loaderTrn, loaderVal, lossFn, optimizer, scheduler, earlyStop, config):
    def train(model, loaderTrn, lossFn, optimizer, scheduler, config):
        model.train()
        totalLoss = 0.0
        for i, (x, xn, bsnAttr, rivAttr, y) in enumerate(loaderTrn):
            x, xn, y = x.squeeze(0), xn.squeeze(0), y.squeeze(0)
            bsnAttr, rivAttr = bsnAttr.squeeze(0), rivAttr.squeeze(0)
            output = model(x, xn, bsnAttr, rivAttr, staIdx=config['data']['spinUp'][0])
            yPred = output['Qr'][config['train']['outBsnIdx'], :].clone().permute(1, 0)
            loss = lossFn(yTrue=y, yPred=yPred, spinUp=config['data']['spinUp'][0])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip'])
            optimizer.step()
            totalLoss += loss.item()
            if i % 2 == 0:
                logging.info(f'Iter {i} of {len(loaderTrn)}: Loss {loss.item():.3f}')


        epochLoss = totalLoss / len(loaderTrn)
        if scheduler is not None:
            scheduler.step(epochLoss)
        return epochLoss

    def valid(model, loaderVal, lossFn, config):
        model.eval()
        with torch.no_grad():
            totalLoss = 0.0
            for x, xn, bsnAttr, rivAttr, y in loaderVal:
                x, xn, y = x.squeeze(0), xn.squeeze(0), y.squeeze(0)
                bsnAttr, rivAttr = bsnAttr.squeeze(0), rivAttr.squeeze(0)
                output = model(x, xn, bsnAttr, rivAttr, staIdx=config['data']['spinUp'][1])
                yPred = output['Qr'][config['train']['outBsnIdx'], :].clone().permute(1, 0)
                loss = lossFn(yTrue=y, yPred=yPred, spinUp=config['data']['spinUp'][1])
                totalLoss += loss.item()
            epochLoss = totalLoss / len(loaderVal)
        return epochLoss

    lossTrnLst, lossValLst = [], []
    for epoch in range(config['train']['epochs']):
        logging.info('*' * 100)
        logging.info('Epoch:{:d}/{:d}'.format(epoch, config['train']['epochs']))
        lossTrn = train(model, loaderTrn, lossFn, optimizer, scheduler, config)
        lossTrnLst.append(lossTrn)
        logging.info(f'Epoch training loss: {lossTrn:.3f}')
        lossVal = valid(model, loaderVal, lossFn, config)
        lossValLst.append(lossVal)
        logging.info(f'Epoch validation loss: {lossVal:.3f}')
        earlyStop(lossVal, model)
        if earlyStop.earlyStop:
            logging.info(f'Early stopping with best loss: {earlyStop.bestLoss: .3f}')
            break
    plotTrnCurve(lossTrnLst, lossValLst, os.path.join(config['out'], 'training_curve.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configFile', type=Union[str, None], default=None,
                        help='the path of configure file determining the hyper-parameters')
    args = parser.parse_args()

    # prepare configures
    if args.configFile is not None:
        with open(args.configFile, 'r') as f:
            config = json.load(f)
    else:
        config = defaultdict()
        # arguments for dataloader
        config['data'] = {'xDataFile': 'data/data.pkl',
                          'flowFile': 'data/streamflow.csv',
                          'periods': [['1960-1-1', '1989-12-31'], ['1990-1-1', '1999-12-31'],
                                      ['2000-1-1', '2019-12-31']],
                          'hydStations': ['TNH', 'MAQ', 'JIM'],
                          'excludeBsnAttrVar': 'slope_min',
                          'spinUp': [365, 365, 365],
                          'seqLen': 1460, 'winSz': 1460, 'mainIdx': 0, 'pad': 9999,
                          'hydStationBsnIdx': {'HHY': 13, 'JIM': 45, 'MET': 53, 'MAQ': 35, 'JUG': 23, 'TNH': 0}}
        # hyper-parameters for training
        if isinstance(config['data']['hydStations'], str):
            config['data']['hydStations'] = [config['data']['hydStations']]
        outBsnIdx = [config['data']['hydStationBsnIdx'][bsn] for bsn in config['data']['hydStations']]
        # get five random seeds using (np.random.uniform(low=0, high=1, size=5) * (10**6)).astype(int)
        # [668823, 759826, 211765, 908331, 808530, 178716]
        config['train'] = {'logLoss': False, 'wLog': 0.25, 'wStationLoss': {'TNH': 1, 'MAQ': 1, 'JIM': 1, 'HHY': 0.5},
                           'patience': 15, 'lr': 0.005, 'clip': 3, 'epochs': 200, 'gpu': True, 'seed': 908331,
                           'outBsnIdx': outBsnIdx, 'dropout': 0.5, 'activFn': 'sigmoid', 'nMul': 16, 'gpu_id': 0,
                           'modelType': 'TL-a'}
        # hyper-parameters for model
        config['staNet'] = {'type': 'ConvMLP',  # ['MLP', 'LSTM', 'LSTMMLP', 'ConvMLP']
                            'inFC': 134, 'hidFC': 128, 'outFC': 7, 'inLSTM': 3, 'hidLSTM': 128, 'outLSTM': 64,
                            'nAttr': 70, 'nMet': 3, 'lenMet': config['data']['spinUp'][0],
                            'nKernel': [10, 5, 1], 'kernelSz': [7, 5, 3], 'stride': [1, 1, 1], 'poolSz': [3, 2, 1]}
        config['dynNet'] = {'type': 'LSTM',  # ['LSTM', 'LSTMCell']
                            'inLSTM': 73, 'hidLSTM': 128, 'outLSTM': 3}
        config['routNet'] = {'inFC': 6, 'hidFC': 32, 'outFC': 2}

    # Configure output file path
    pretrainedModelPt = './data/model_Ep50.pt'
    now = time.strftime('%m%d-%H%M', time.localtime())
    staType = config['staNet']['type']
    if staType == 'MLP':
        staSz = [config['staNet']['inFC'], config['staNet']['hidFC'], config['staNet']['outFC']]
    elif staType == 'LSTM':
        staSz = [config['staNet']['inLSTM'], config['staNet']['hidLSTM'], config['staNet']['outLSTM']]
    elif staType == 'LSTMMLP':
        staSz = [config['staNet']['inFC'], config['staNet']['hidFC'], config['staNet']['outFC'],
                 config['staNet']['inLSTM'], config['staNet']['hidLSTM'], config['staNet']['outLSTM']]
    else:
        staSz = [config['staNet']['nAttr'], config['staNet']['nMet'], config['staNet']['hidFC'],
                 config['staNet']['outFC'], config['staNet']['lenMet'], config['staNet']['nKernel'],
                 config['staNet']['kernelSz'], config['staNet']['stride'], config['staNet']['poolSz']]
    dynType = config['dynNet']['type']
    dynSz = [config['dynNet']['inLSTM'], config['dynNet']['hidLSTM'], config['dynNet']['outLSTM']]
    routSz = [config['routNet']['inFC'], config['routNet']['hidFC'], config['routNet']['outFC']]
    seed = config['train']['seed']
    hydStation = config['data']['hydStations']
    nMul = config['train']['nMul']
    seqLen = config['data']['seqLen']
    modelType = config['train']['modelType']
    config['out'] = f"./checkpoints/{modelType}/seed_{seed}_nMul_{nMul}_seqLen_{seqLen}_out_{'_'.join(hydStation)}_staTyp_{staType}_staSz_" \
                    f"{'_'.join([str(sz) if isinstance(sz, int) else '-'.join(str(e) for e in sz) for sz in staSz])}" \
                    f"_dynTyp_{dynType}_dynSz_{'_'.join([str(sz) for sz in dynSz])}_routSz_" \
                    f"{'_'.join([str(sz) for sz in routSz])}_t_{now}"
    Path(config['out']).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(config['out'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # fix the random seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # configure log file
    logFile = os.path.join(config['out'], 'log.txt')
    if os.path.exists(logFile):
        os.remove(logFile)
    logging.basicConfig(filename=logFile, level=logging.INFO, format='%(asctime)s: %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    # determine the device
    device = torch.device(f"cuda:{config['train']['gpu_id']}" if config['train']['gpu'] and torch.cuda.is_available() else 'cpu')
    logging.info(f'{device} is used in training.')
    logging.info(f'The output path is {config["out"]}')

    # get dataloaders
    loader = DataLoader(xdataFile=config['data']['xDataFile'], flowFile=config['data']['flowFile'],
                        periods=config['data']['periods'], spinUp=config['data']['spinUp'],
                        seqLen=config['data']['seqLen'], winSize=config['data']['winSz'], device=device,
                        hydStation=config['data']['hydStations'], excludeBsnAttrVar=config['data']['excludeBsnAttrVar'])

    # configure model
    model = Net(staType=staType, staSz=staSz, dynType=dynType, dynSz=dynSz, routSz=routSz,
                routOrder=loader.routOrder, area=loader.area, upIdx=loader.dataAll['rivUpIdx'],
                mainIdx=config['data']['mainIdx'], pad=config['data']['pad'], dropout=config['train']['dropout'],
                activFn=config['train']['activFn'], device=device, nMul=nMul)
    model = model.to(device)
    newStateDict = model.state_dict()

    # determine which parameters are frozen during training
    assert config['train']['modelType'] in ['TL-a', 'TL-b', 'TL-c', 'TL-d']
    # Define a list of parameter names to be frozen
    if config['train']['modelType'].startswith('TL'):
        # load pretrained Camels model
        preModel = torch.load(pretrainedModelPt, map_location=device)
        preStateDict = preModel.state_dict()
        for name, param in preStateDict.items():
            if name not in ['dPL.staNet.fc.0.weight', 'dPL.staNet.fc.0.bias', 'dPL.dynNet.fcIn.0.weight',
                            'dPL.dynNet.fcIn.0.bias']:
                newStateDict[name[4:]].copy_(param)
        model.load_state_dict(newStateDict)

    if config['train']['modelType'] == 'TL-a':
        paraToFreeze = ['dynNet.LSTM.lstm.weight_ih_l0', 'dynNet.LSTM.lstm.weight_hh_l0',
                        'dynNet.LSTM.lstm.bias_ih_l0',  'dynNet.LSTM.lstm.bias_hh_l0',
                        'staNet.convLayers.CnnLayer1.cnn1d.weight', 'staNet.convLayers.CnnLayer1.cnn1d.bias',
                        'staNet.convLayers.CnnLayer2.cnn1d.weight', 'staNet.convLayers.CnnLayer2.cnn1d.bias',
                        'staNet.convLayers.CnnLayer3.cnn1d.weight', 'staNet.convLayers.CnnLayer3.cnn1d.bias']
    elif config['train']['modelType'] == 'TL-b':
        paraToFreeze = ['dynNet.LSTM.lstm.weight_ih_l0', 'dynNet.LSTM.lstm.weight_hh_l0',
                        'dynNet.LSTM.lstm.bias_ih_l0',  'dynNet.LSTM.lstm.bias_hh_l0']
    elif config['train']['modelType'] == 'TL-c':
        paraToFreeze = ['staNet.convLayers.CnnLayer1.cnn1d.weight', 'staNet.convLayers.CnnLayer1.cnn1d.bias',
                        'staNet.convLayers.CnnLayer2.cnn1d.weight', 'staNet.convLayers.CnnLayer2.cnn1d.bias',
                        'staNet.convLayers.CnnLayer3.cnn1d.weight', 'staNet.convLayers.CnnLayer3.cnn1d.bias']
    else:
        paraToFreeze = []

    # Freeze the specified parameters
    for name, param in model.named_parameters():
        if any(paraName in name for paraName in paraToFreeze):
            param.requires_grad = False

    # re-train the distributed TL model
    wStation = [config['train']['wStationLoss'][station] for station in config['data']['hydStations']]
    lossFn = nseLoss(logLossOpt=config['train']['logLoss'], wLog=config['train']['wLog'], wStation=wStation)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    earlyStop = earlyStopping(savePath=os.path.join(config['out'], 'model.pt'), patience=config['train']['patience'],
                              delta=0.0002)
    trainModel(model=model, loaderTrn=loader.loaderTrn, loaderVal=loader.loaderVal, lossFn=lossFn, optimizer=optimizer,
               scheduler=scheduler, earlyStop=earlyStop, config=config)

    # get and save simulations
    model.load_state_dict(torch.load(os.path.join(config['out'], 'model.pt')))
    saveSimulation(model=model, loaderVal=loader.loaderVal, loaderTst=loader.loaderTst, outPath='simulation.pkl',
                   config=config)
    # os._exit(0)

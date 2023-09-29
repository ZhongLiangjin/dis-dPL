import logging
import time
from pathlib import Path
from hydroDL.model import rnn
import torch.backends.cudnn as cudnn
import os
import numpy as np
import torch
from collections import OrderedDict
import random
import json
from utils.trainTools import nseLoss, earlyStopping, plotTrnCurve, saveSimulation
from loader import DataLoader


def trainModel(model, loaderTrn, loaderVal, lossFn, optimizer, scheduler, earlyStop, config):
    def train(model, loaderTrn, lossFn, optimizer, scheduler, config):
        model.train()
        totalLoss = 0.0
        for i, (xTrain, zTrain, y) in enumerate(loaderTrn):
            xTrain, zTrain, y = xTrain.squeeze(0), zTrain.squeeze(0), y.squeeze(0)
            output = model(xTrain, zTrain)
            yPred = output[:, :, 0].clone()
            loss = lossFn(yTrue=y, yPred=yPred, spinUp=config['data']['spinUp'])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), config['train']['clip'])
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
            for i, (xVal, zVal, y) in enumerate(loaderVal):
                xVal, zVal, y = xVal.squeeze(0), zVal.squeeze(0), y.squeeze(0)
                output = model(xVal, zVal)
                yPred = output[:, :, 0].clone()
                loss = lossFn(yTrue=y, yPred=yPred, spinUp=config['data']['spinUp'])
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
    # set hyperparameters
    modelType = 'TL-d'  # one of ['TL-a', 'TL-b', 'TL-c', 'TL-d', 'dPL']
    nMul = 16  # 16 components
    spinUp = 730  # for each training sample, to use BUFFTIME days to warm up the states.
    routing = True  # Whether to use the routing module for simulated runoff
    compRout = False  # True is doing routing for each component
    compWts = False  # True is using weighted average for components; False is the simple mean

    hydStations = ['TNH', 'MAQ', 'JIM']
    periods = [['1960-1-1', '1989-12-31'], ['1990-1-1', '1999-12-31'], ['2000-1-1', '2019-12-31']]
    xTrainDataFile = './data/xdata_6basins.pkl'
    xTestDataFile = './data/xdata_69basins.pkl'
    flowFile = './data/streamflow.csv'
    pretrainedModelPt = './data/model_Ep50.pt'
    excludeBsnAttrVar = 'slope_min'
    seqLen = 2190

    lr = 0.001
    epochs = 200
    logLoss = False  # calculate the loss of log streamflow
    wLog = 0.25  # weight of log loss
    seed = 178716  # [668823, 759826, 211765, 908331, 808530, 178716]
    wStationLoss = {'TNH': 1, 'MAQ': 1, 'JIM': 1}  # loss weights for different stations
    patience = 15
    clip = 0.5
    gpuId = 0

    # fix the random seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # get dataloaders
    device = torch.device(f'cuda:{gpuId}' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(xTestDataFile=xTestDataFile, xTrainDataFile=xTrainDataFile, flowFile=flowFile, device=device,
                        periods=periods, spinUp=spinUp, seqLen=seqLen, winSize=seqLen, hydStation=hydStations,
                        excludeBsnAttrVar=['slope_min'])
    nInv = loader.loaderTrn.dataset.xn.shape[-1] + loader.loaderTrn.dataset.bsnAttr.shape[-1]

    # config model
    staNet = OrderedDict(hidFC=256, outFC=5, nAttr=70, nMet=3, spinUp=spinUp,
                         nKernel=[10, 5, 1], kernelSz=[7, 5, 3], stride=[1, 1, 1], poolSz=[3, 2, 1])
    dynNet = OrderedDict(inLSTM=73, hidLSTM=256, outLSTM=3)
    optData = OrderedDict(xTrainDataFile=xTrainDataFile, xTestDataFile=xTestDataFile, flowFile=flowFile,
                          pretrainedModelPt=pretrainedModelPt, hydStations=hydStations, periods=periods,
                          spinUp=spinUp, excludeBsnAttrVar=excludeBsnAttrVar, seqLen=seqLen)
    optTrain = OrderedDict(lr=lr, epochs=epochs, logLoss=logLoss, wLog=wLog, seed=seed, wStationLoss=wStationLoss,
                           patience=patience, clip=clip, gpu=True)
    now = time.strftime('%m%d-%H%M', time.localtime())
    outPath = f"checkpoints/{modelType}/seed_{seed}_out_{'_'.join(hydStations)}_logLoss_{logLoss}_t_{now}"
    config = OrderedDict(staNet=staNet, dyNet=dynNet, data=optData, train=optTrain, nMul=nMul, out=outPath)
    Path(outPath).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(outPath, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # configure log file
    logFile = os.path.join(outPath, 'log.txt')
    if os.path.exists(logFile):
        os.remove(logFile)
    logging.basicConfig(filename=logFile, level=logging.INFO, format='%(asctime)s: %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    # define and load model
    staSz = [staNet['nAttr'], staNet['nMet'], staNet['hidFC'], staNet['outFC'], staNet['spinUp'], staNet['nKernel'],
             staNet['kernelSz'], staNet['stride'], staNet['poolSz']]
    dynSz = [dynNet['inLSTM'], dynNet['hidLSTM'], dynNet['outLSTM']]
    model = rnn.MultiInv_EXPHYDROTDModel(staSz, dynSz, nMul, device=device)
    model = model.to(device)
    newStateDict = model.state_dict()

    # determine which parameters are frozen during training
    assert modelType in ['TL-a', 'TL-b', 'TL-c', 'TL-d', 'dPL']
    # Define a list of parameter names to be frozen
    if modelType.startswith('TL'):
        # load pretrained Camels model
        preModel = torch.load(pretrainedModelPt, map_location=device)
        preStateDict = preModel.state_dict()
        for name, param in preStateDict.items():
            if (name not in ['dPL.staNet.fc.0.weight', 'dPL.staNet.fc.0.bias', 'dPL.dynNet.fcIn.0.weight',
                             'dPL.dynNet.fcIn.0.bias']) & (name in newStateDict):
                newStateDict[name].copy_(param)
        model.load_state_dict(newStateDict)

    if modelType == 'TL-a':
        paraToFreeze = ['dPL.dynNet.LSTM.lstm.weight_ih_l0', 'dPL.dynNet.LSTM.lstm.weight_hh_l0',
                        'dPL.dynNet.LSTM.lstm.bias_ih_l0',  'dPL.dynNet.LSTM.lstm.bias_hh_l0',
                        'dPL.staNet.convLayers.CnnLayer1.cnn1d.weight', 'dPL.staNet.convLayers.CnnLayer1.cnn1d.bias',
                        'dPL.staNet.convLayers.CnnLayer2.cnn1d.weight', 'dPL.staNet.convLayers.CnnLayer2.cnn1d.bias',
                        'dPL.staNet.convLayers.CnnLayer3.cnn1d.weight', 'dPL.staNet.convLayers.CnnLayer3.cnn1d.bias']
    elif modelType == 'TL-b':
        paraToFreeze = ['dPL.dynNet.LSTM.lstm.weight_ih_l0', 'dPL.dynNet.LSTM.lstm.weight_hh_l0',
                        'dPL.dynNet.LSTM.lstm.bias_ih_l0',  'dPL.dynNet.LSTM.lstm.bias_hh_l0']
    elif modelType == 'TL-c':
        paraToFreeze = ['dPL.staNet.convLayers.CnnLayer1.cnn1d.weight', 'dPL.staNet.convLayers.CnnLayer1.cnn1d.bias',
                        'dPL.staNet.convLayers.CnnLayer2.cnn1d.weight', 'dPL.staNet.convLayers.CnnLayer2.cnn1d.bias',
                        'dPL.staNet.convLayers.CnnLayer3.cnn1d.weight', 'dPL.staNet.convLayers.CnnLayer3.cnn1d.bias']
    else:
        paraToFreeze = []

    # Freeze the specified parameters
    for name, param in model.named_parameters():
        if any(paraName in name for paraName in paraToFreeze):
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    # define loss function
    wStation = [wStationLoss[station] for station in hydStations]
    lossFn = nseLoss(logLossOpt=False, wLog=0.25, wStation=wStation)
    earlyStop = earlyStopping(savePath=os.path.join(outPath, 'model.pt'), patience=patience, delta=0.0002)

    logging.info(f'{device} is used for training.')
    trainModel(model=model, loaderTrn=loader.loaderTrn, loaderVal=loader.loaderVal, lossFn=lossFn, optimizer=optimizer,
               scheduler=scheduler, earlyStop=earlyStop, config=config)
    saveSimulation(model=model, loaderVal=loader.loaderVal, loaderTst=loader.loaderTst, outPath='simulation.pkl',
                   config=config)
    logging.shutdown()










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
    for folder in os.listdir('./checkpoints/TL-d'):
        path = f'./checkpoints/TL-d/{folder}'
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)

        seed = config['train']['seed']
        # fix the random seed
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

        # get dataloaders
        device = torch.device(f'cuda:0' if ((torch.cuda.is_available()) & (config['train']['gpu'])) else 'cpu')
        xTrainDataFile = config['data']['xTrainDataFile']
        xTestDataFile = config['data']['xTestDataFile']
        flowFile = config['data']['flowFile']
        periods = config['data']['periods']
        spinUp = config['data']['spinUp']
        hydStations = config['data']['hydStations']
        seqLen = config['data']['seqLen']


        loader = DataLoader(xTestDataFile=xTestDataFile, xTrainDataFile=xTrainDataFile, flowFile=flowFile, device=device,
                            periods=periods, spinUp=spinUp, seqLen=seqLen, winSize=seqLen, hydStation=hydStations,
                            excludeBsnAttrVar=['slope_min'])

        # config model
        staNet, dynNet, outPath, nMul = config['staNet'], config['dyNet'], config['out'], config['nMul']

        # define and load model
        staSz = [staNet['nAttr'], staNet['nMet'], staNet['hidFC'], staNet['outFC'], staNet['spinUp'], staNet['nKernel'],
                 staNet['kernelSz'], staNet['stride'], staNet['poolSz']]
        dynSz = [dynNet['inLSTM'], dynNet['hidLSTM'], dynNet['outLSTM']]
        model = rnn.MultiInv_EXPHYDROTDModel(staSz, dynSz, nMul, device=device)
        model = model.to(device)
        model.load_state_dict(torch.load(f'{outPath}/model.pt', map_location=device))

        saveSimulation(model=model, loaderVal=loader.loaderVal, loaderTst=loader.loaderTst, outPath='simulation.pkl',
                       config=config)
        print(folder)











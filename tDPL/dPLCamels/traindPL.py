from hydroDL import master, utils
from hydroDL.data import camels
from hydroDL.master import default
from hydroDL.model import rnn, crit, train
import os
import numpy as np
import torch
from collections import OrderedDict
import random
import json
from utils.trainTools import nseLoss
import time

## fix the random seeds for reproducibility
randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## GPU setting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

forType = 'daymet'
# for Type defines which forcing in CAMELS to use: 'daymet', 'nldas', 'maurer'
## Set hyperparameters
EPOCH = 50  # total epoches to train the mode
BATCH_SIZE = 100
RHO = 730
saveEPOCH = 5
Ttrain = [19801001, 19951001]  # Training period
Tinv = [19801001, 19951001]  # Inversion period for historical forcings
spinUp = 730  # for each training sample, to use BUFFTIME days to warm up the states.
Nmul = 16  # Multi-component model. How many parallel HBV components to use. 1 means the original HBV.

rootDatabase = './Camels'  # CAMELS dataset root directory
camels.initcamels(
    rootDatabase)  # initialize camels module-scope variables in camels.py (dirDB, gageDict) to read basin info
rootOut = './output'  # Model output root directory

# load CAMELS basin information
gageinfo = camels.gageDict
hucinfo = gageinfo['huc']
gageid = gageinfo['id']
gageidLst = gageid.tolist()

TrainLS = gageidLst  # all basins
TrainInd = [gageidLst.index(j) for j in TrainLS]
TestLS = gageidLst
TestInd = [gageidLst.index(j) for j in TestLS]
gageDic = {'TrainID': TrainLS, 'TestID': TestLS}

TtrainLoad = Ttrain
TinvLoad = Tinv

## prepare input data
## load camels dataset
if forType is 'daymet':
    varF = ['prcp', 'tmean']
    varFInv = ['prcp', 'tmean']
else:
    varF = ['prcp', 'tmax']  # For CAMELS maurer and nldas forcings, tmax is actually tmean
    varFInv = ['prcp', 'tmax']

# the attributes used to learn parameters
attrnewLst = ['p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
              'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
              'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
              'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
              'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
              'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability']

optData = default.optDataCamels  # a default dictionary for logging, updated below
# Update the training period and variables
optData = default.update(optData, tRange=TtrainLoad, varT=varFInv, varC=attrnewLst, subset=TrainLS, forType=forType)

# for HBV model training inputs
dfTrain = camels.DataframeCamels(tRange=TtrainLoad, subset=TrainLS, forType=forType)
forcUN = dfTrain.getDataTs(varLst=varF, doNorm=False, rmNan=False)
obsUN = dfTrain.getDataObs(doNorm=False, rmNan=False, basinnorm=False)

# for dPL inversion data, inputs of gA
dfInv = camels.DataframeCamels(tRange=TinvLoad, subset=TrainLS, forType=forType)
forcInvUN = dfInv.getDataTs(varLst=varFInv, doNorm=False, rmNan=False)
attrsUN = dfInv.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)

# Unit transformation, discharge obs from ft3/s to mm/day
areas = gageinfo['area'][TrainInd]  # unit km2
temparea = np.tile(areas[:, None, None], (1, obsUN.shape[1], 1))
obsUN = (obsUN * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3  # transform to mm/day

# load potential ET calculated by hargreaves method
varLstNL = ['PEVAP']
usgsIdLst = gageid
if forType == 'maurer':
    tPETRange = [19800101, 20090101]
else:
    tPETRange = [19800101, 20150101]
tPETLst = utils.time.tRange2Array(tPETRange)

# Modify this as the directory where you put PET
PETDir = rootDatabase + '/pet_harg/' + forType + '/'
ntime = len(tPETLst)
PETfull = np.empty([len(usgsIdLst), ntime, len(varLstNL)])
for k in range(len(usgsIdLst)):
    dataTemp = camels.readcsvGage(PETDir, usgsIdLst[k], varLstNL, ntime)
    PETfull[k, :, :] = dataTemp

TtrainLst = utils.time.tRange2Array(TtrainLoad)
TinvLst = utils.time.tRange2Array(TinvLoad)
C, ind1, ind2 = np.intersect1d(TtrainLst, tPETLst, return_indices=True)
PETUN = PETfull[:, ind2, :]
PETUN = PETUN[TrainInd, :, :]  # select basins
C, ind1, ind2inv = np.intersect1d(TinvLst, tPETLst, return_indices=True)
PETInvUN = PETfull[:, ind2inv, :]
PETInvUN = PETInvUN[TrainInd, :, :]

# process data, do normalization and remove nan
series_inv = np.concatenate([forcInvUN, PETInvUN], axis=2)
seriesvarLst = varFInv + ['pet']
# calculate statistics for normalization and saved to a dictionary
statDict = camels.getStatDic(attrLst=attrnewLst, attrdata=attrsUN, seriesLst=seriesvarLst, seriesdata=series_inv)
# normalize data
attr_norm = camels.transNormbyDic(attrsUN, attrnewLst, statDict, toNorm=True)
attr_norm[np.isnan(attr_norm)] = 0.0
series_norm = camels.transNormbyDic(series_inv, seriesvarLst, statDict, toNorm=True)
series_norm[np.isnan(series_norm)] = 0.0

# prepare the inputs
zTrainIn = series_norm
xTrainIn = np.concatenate([forcUN, PETUN], axis=2)  # used as HBV forcing
xTrainIn[np.isnan(xTrainIn)] = 0.0
yTrainIn = obsUN

forcTuple = (xTrainIn, zTrainIn)
attrs = attr_norm

## Train the model
# define loss function
alpha = 0.25  # a weight for RMSE loss to balance low and peak flow
optLoss = default.update(default.optLossComb, name='hydroDL.model.crit.RmseLossComb', weight=alpha)

# define training options
optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH, saveEpoch=saveEPOCH)
now = time.strftime('%m%d-%H%M', time.localtime())


# dict only for logging
staNet = OrderedDict(hidFC=256, outFC=5, nAttr=attrs.shape[-1], nMet=xTrainIn.shape[-1], spinUp=spinUp,
                     nKernel=[10, 5, 1], kernelSz=[7, 5, 3], stride=[1, 1, 1], poolSz=[3, 2, 1])
dynNet = OrderedDict(inLSTM=attrs.shape[-1] + xTrainIn.shape[-1], hidLSTM=256, outLSTM=3)
optModel = OrderedDict(staNet=staNet, dyNet=dynNet, Trainbuff=spinUp, rho=RHO, nMul=Nmul)

# define and load model
staSz = [staNet['nAttr'], staNet['nMet'], staNet['hidFC'], staNet['outFC'], staNet['spinUp'], staNet['nKernel'],
         staNet['kernelSz'], staNet['stride'], staNet['poolSz']]
dynSz = [dynNet['inLSTM'], dynNet['hidLSTM'], dynNet['outLSTM']]

out = f"./checkpoints/seed_{randomseed}_staNet_{'_'.join([str(sz) if isinstance(sz, int) else '-'.join(str(e) for e in sz) for sz in staSz])}" \
      f"_dynSz_{'_'.join([str(sz) for sz in dynSz])}_t_{now}"  # output folder to save results

model = rnn.MultiInv_EXPHYDROTDModel(staSz, dynSz, Nmul, device=device)
model = model.to(device)
lossFun = nseLoss(logLossOpt=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.99))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

# Wrap up all the training configurations to one dictionary in order to save into "out" folder as logging
masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
master.writeMasterFile(masterDict)
# log statistics for normalization
statFile = os.path.join(out, 'statDict.json')
with open(statFile, 'w') as fp:
    json.dump(statDict, fp, indent=4)
# Train the model
trainedModel = train.trainModel(
    model,
    forcTuple,
    yTrainIn,
    attrs,
    lossFun,
    optimizer,
    scheduler,
    nEpoch=EPOCH,
    miniBatch=[BATCH_SIZE, RHO],
    saveEpoch=saveEPOCH,
    saveFolder=out,
    bufftime=spinUp)

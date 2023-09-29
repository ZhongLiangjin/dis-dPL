from collections import defaultdict
from typing import List, Union
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, staType: str, staSz: List[Union[int, List[int]]], dynType: str, dynSz: List[int],
                 routSz: List[Union[int, List[int]]], routOrder: np.ndarray,
                 area: np.ndarray, upIdx: List[int], mainIdx: int = 0, pad: int = 9999, dropout: float = 0.5,
                 activFn: str = 'sigmoid', device: Union[str, torch.device] = 'cpu'):
        """
        :param staType: One of value in ['MLP', 'LSTM', 'LSTMMLP', 'ConvMLP'], deciding which model to get static
            parameters.
        :param staSz: A list consisting of int or List[int] to define the size of network to learn static parameter with
            a shape of [inFC, hidFC, outFC, inLSTM, hidLSTM, outLSTM].
        :param dynType: One of value in ['LSTM', 'LSTMCell'], deciding which model to get dynamic parameters.
        :param dynSz: A list consisting of int to define the size of network to learn dynamic parameter with a shape of
            [inLSTM, hidLSTM, outLSTM].
        :param routSz: A list consisting of int or List[int] to define the size of network to learn river routing
            parameter with a shape of [inFC, hidFC, outFC].
        :param routOrder: an array containing the channel routing order from upstream to downstream.
        :param area: area of sub-basins (km2) with a shape of [N, ].
        :param upIdx: a list used to identify the upstream sub-basin of each river reach with a shape of [N_river, ].
        :param mainIdx: the row number of mainstream sub-basins' routing order in the routOrder.
        :param pad: the padded value in the routOrder to indicate the end of the river reach.
        :param activFn: activation function used to get parameters.
        """
        super(Net, self).__init__()
        assert staType in ['MLP', 'LSTM', 'LSTMMLP', 'ConvMLP']
        assert dynType in ['LSTM', 'LSTMCell']

        self.staType = staType
        self.dynType = dynType
        self.activFn = activFn
        self.device = device

        # define the network for static parameter learning
        if staType == 'MLP':
            staInFc, staHidFc, staOutFc = staSz
            self.staNet = StaParasMLP(inSz=staInFc, hidSz=staHidFc, outSz=staOutFc, dropout=dropout)
        elif staType == 'LSTM':
            staInLSTM, staHidLSTM, staOutLSTM = staSz
            self.staNet = StaParasLSTM(inLSTM=staInLSTM, hidLSTM=staHidLSTM, outLSTM=staOutLSTM,
                                       dropout=dropout, device=device)
        elif staType == 'LSTMMLP':
            staInFc, staHidFc, staOutFc, staInLSTM, staHidLSTM, staOutLSTM = staSz
            self.staNet = StaParasLSTMMLP(inLSTM=staInLSTM, hidLSTM=staHidLSTM, outLSTM=staOutLSTM,
                                          inFC=staInFc, hidFC=staHidFc, outFC=staOutFc, dropout=dropout, device=device)
        else:
            nAttr, nMet, hidFC, outFC, lenMet, nKernel, kernelSz, stride, poolSz = staSz
            self.staNet = StaParasConv1DMLP(nx=nAttr, nz=nMet, ny=outFC, inLen=lenMet, hidden=hidFC, nKernel=nKernel,
                                            kernelSz=kernelSz, stride=stride, poolKerSz=poolSz, dropout=dropout)

        # define the network for dynamic parameter learning
        dynInLSTM, dynHidLSTM, dynOutLSTM = dynSz
        if dynType == 'LSTM':
            self.dynNet = DynParasLSTM(inLSTM=dynInLSTM, hidLSTM=dynHidLSTM, outLSTM=dynOutLSTM,
                                       dropout=dropout, device=device)
        else:
            self.dynNet = DynParasLSTMCell(inLSTM=dynInLSTM, hidLSTM=dynHidLSTM, outLSTM=dynOutLSTM, dropout=dropout,
                                           device=device)

        # define the network for river routing parameters learning
        routInFC, routHidFC, routOutFC = routSz
        self.routNet = RoutMLP(inSize=routInFC, hidSize=routHidFC, outSize=routOutFC, dropout=dropout, device=device)

        # define the FLEX model
        self.hydromodel = ExpHYDROCell(routOrder, area, upIdx, mainIdx, pad, device)

    def ScaleParas(self, paras: torch.Tensor, paraType: str):
        """
        Constrain parameters into physical ranges.

        Reference
        ---------
        Gao, H., Hrachowitz, M., Fenicia, F., Gharari, S., & Savenije, H. H. G. (2014). Testing the realism of a
        topography-driven model (FLEX-Topo) in the nested catchments of the Upper Heihe, China. Hydrology and Earth
        System Sciences, 18(5), 1895-1915. http://doi.org/10.5194/hess-18-1895-2014

        Savenije, H. H. G. (1997). Determination of evaporation from a catchment water balance at a monthly time scale.
        Hydrol. Earth Syst. Sci., 1(1), 93-100. http://doi.org/10.5194/hess-1-93-1997

        Trotter, L., Knoben, W. J. M., Fowler, K. J. A., Saft, M., & Peel, M. C. (2022). Modular Assessment of
        Rainfall–Runoff Models Toolbox (MARRMoT) v2.1: an object-oriented implementation of 47 established hydrological
        models for improved speed and readability. Geosci. Model Dev., 15(16), 6359-6369.
        http://doi.org/10.5194/gmd-15-6359-2022
        """
        assert paraType in ['static', 'dynamic', 'routing']
        assert self.activFn in ['sigmoid', 'tanh']
        if self.activFn == 'sigmoid':
            paras = torch.sigmoid(paras)
        else:
            paras = (torch.tanh(paras) + 1) / 2  # scale to the range of 0~1

        scaleParas = torch.zeros(paras.size(), dtype=torch.float32).to(self.device)
        scaleLt = {
            # parTMIN, parDDF, parTMAX, parQMAX, parA, and parB, parSMAX
            'static': [[-2, 1], [1, 8], [-1, 2], [10, 50], [0, 2.9], [0, 6.5], [100, 1500]],
            # parF, parALPHA and parBETA
            'dynamic': [[0, 0.1], [0, 1], [0.5, 2]]
        }
        if paraType in ['static', 'dynamic']:
            if paras.dim() == 2:
                for idx in range(len(scaleLt[paraType])):
                    scaleParas[:, idx] = scaleLt[paraType][idx][0] + paras[:, idx] * (scaleLt[paraType][idx][1] -
                                                                                      scaleLt[paraType][idx][0])
            elif paras.dim() == 3:
                for idx in range(len(scaleLt[paraType])):
                    scaleParas[:, :, idx] = scaleLt[paraType][idx][0] + paras[:, :, idx] * (scaleLt[paraType][idx][1] -
                                                                                            scaleLt[paraType][idx][0])
            else:
                raise ValueError("Input tensor must be 2-dimensional or 3-dimensional.")

        else:
            if paras.dim() == 2:
                scaleParas[:, 0:1] = paras[:, 0:1] * 0.5  # constrain parX in Muskingum into [0, 0.5]
                scaleParas[:, 1:2] = 1 + paras[:, 1:2] * (1 / 2 / paras[:, 0:1] - 1)  # constrain parK with 2𝐾𝑋<Δt≤𝐾

            elif paras.dim() == 3:
                scaleParas[:, :, 0:1] = paras[:, :, 0:1] * 0.5  # constrain parX in Muskingum into [0, 0.5]
                scaleParas[:, :, 1:2] = 1 + paras[:, :, 1:2] * (
                        1 / 2 / paras[:, :, 0:1] - 1)  # constrain parK with 2𝐾𝑋<Δt≤𝐾

            else:
                raise ValueError("Input tensor must be 2-dimensional or 3-dimensional.")

        return scaleParas

    def forward(self, x: torch.Tensor, xn: torch.Tensor, bsnAttr: torch.Tensor, rivAttr: torch.Tensor,
                staIdx: int = -1, mode: str = 'train'):
        """
        :param staIdx: the index to get static parameters from LSTM generated sequence.
        :param x: input for hydrological model (pr, tas, pet)
        :param xn: normalized x
        :param bsnAttr: basin attributes with a shape of [N-basins, F]
        :param rivAttr: river attributes with a shape of [N-rivers, F]
        :param mode: str, determine to output which hydrological variables. Output 'Qf' and 'Qs' for 'train' mode,
            and 'Qf', 'Qs', 'Sw', 'E', 'Sult', 'Sust', 'Sslt', 'Ssst' for 'analyse' mode.

        """
        assert mode in ['train', 'analyse']
        # get static parameters
        z = torch.cat((xn, bsnAttr.unsqueeze(1).repeat(1, xn.size(1), 1)), dim=-1)
        staParasTemp = self.staNet(z[:, :staIdx, :])
        staParas = self.ScaleParas(staParasTemp, paraType='static')
        # get routing parameters
        routParasTemp = self.routNet(rivAttr)
        routParas = self.ScaleParas(routParasTemp, paraType='routing')
        # get dynamic parameters
        if self.dynType == 'LSTM':
            dynParasTemp = self.dynNet(z)
            dynParas = self.ScaleParas(dynParasTemp, paraType='dynamic')
        else:
            dynParas = None

        # hydrological model simulation
        outVars = ['Qb', 'Qs'] if mode == 'train' else ['Qb', 'Qs', 'Swt', 'E', 'Sslt', 'Ssst']
        R0, S0, SD0 = self.hydromodel.init_model(R0=None, S0=None, SD0=None)  # initiate storages
        hidden = None
        output = defaultdict(list)
        for t in range(z.shape[1]):
            # get parameters
            if self.dynType == 'LSTM':
                dynParas_t = dynParas[:, t, :].clone()
            else:
                temDynParas, hidden = self.dynNet(torch.cat((z[:, t, :], SD0), dim=-1), hidden)
                dynParas_t = self.ScaleParas(temDynParas, paraType='dynamic')
            cellOut = self.hydromodel(x[:, t, :], S0, staParas, dynParas_t)
            for key in outVars:
                output[key].append(cellOut[key])
            # update initial state
            S0 = torch.stack([cellOut[k] for k in ['Swt', 'Sslt', 'Ssst']], dim=-1)
            SD0 = torch.stack([cellOut[k] for k in ['Sslt', 'Ssst']], dim=-1)

        # stack to [N, L]
        output = {key: torch.stack(val, -1) for key, val in output.items() if key in outVars}

        # calculate hill slope routing
        parA, parB = staParas[:, 4:5].repeat(1, z.shape[1]), staParas[:, 5:6].repeat(1, z.shape[1])
        output['Qs'] = self.hydromodel.hillSlopeRouting(output['Qs'], parA, parB, tmax=15)

        # calculate channel routing
        parX, parK = routParas[:, 0:1].repeat(1, z.shape[1]), routParas[:, 1:].repeat(1, z.shape[1])
        output['Qr'] = self.hydromodel.channelRouting(output['Qb'] + output['Qs'], R0, parX, parK)

        return output


class DynParasLSTMCell(nn.Module):
    """
    Use lstm cell to get dynamic parameters with forcing, reservoir storages, and attributes.
    """

    def __init__(self, inLSTM: int, hidLSTM: int, outLSTM: int, dropout: float = 0.5,
                 device: Union[str, torch.device] = 'cpu'):
        super(DynParasLSTMCell, self).__init__()
        self.device = device
        self.hidLSTM = hidLSTM

        self.bn = nn.BatchNorm1d(inLSTM)
        self.fcIn = nn.Sequential(
            nn.Linear(inLSTM, hidLSTM),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.LSTMCell = nn.LSTMCell(hidLSTM, hidLSTM, device=device)
        self.fcOut = nn.Linear(hidLSTM, outLSTM)

    def forward(self, x, hidden=None):
        """
        :param x: consists of forcing, reservoir storages at last timestep t-1, and attributes with a shape of [N, F]
        """
        x = self.bn(x)
        x = self.fcIn(x)
        h_1, c_1 = self.LSTMCell(x, hidden)
        out = self.fcOut(h_1)
        return out, (h_1, c_1)


class DynParasLSTM(nn.Module):
    """
    Use LSTM to learn the dynamic parameters from forcing and attributes.
    """

    def __init__(self, inLSTM: int, hidLSTM: int, outLSTM: int, dropout: float = 0.5,
                 device: Union[str, torch.device] = 'cpu'):
        super(DynParasLSTM, self).__init__()
        self.device = device
        self.hidLSTM = hidLSTM

        self.fcIn = nn.Sequential(
            nn.Linear(inLSTM, hidLSTM),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.LSTM = LSTM(hidLSTM, hidLSTM, device=device)
        self.fcOut = nn.Linear(hidLSTM, outLSTM)

    def forward(self, x, hidden=None):
        """
        :param x: consists of forcing and attributes with a shape of [N, L, F]
        """
        x = self.fcIn(x)
        hidden = self.LSTM.init_hidden(x.shape[0]) if hidden is None else hidden
        x, hidden = self.LSTM(x, hidden)
        out = self.fcOut(x)
        return out


class StaParasMLP(nn.Module):
    """  Only use FC layers to learn the static parameters from static attributes. """

    def __init__(self, inSz: int, hidSz: Union[int, List[int]], outSz: int, dropout: float = 0.5):
        super(StaParasMLP, self).__init__()
        layers = []
        inFeatures = inSz
        if isinstance(hidSz, int):
            hidSz = [hidSz]
        for i in range(len(hidSz)):
            outFeatures = hidSz[i]
            layers.append(nn.Linear(inFeatures, outFeatures))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            inFeatures = outFeatures
        layers.append(nn.Linear(inFeatures, outSz))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: consists of forcing and static attributes, where attributes keep the same along dimension L,
            with a shape of (N, L, F).
        """
        x = x[:, -1, 4:]
        out = self.net(x)
        return out


class StaParasLSTMMLP(nn.Module):
    """
    Use LSTM to extract additional features from historical forcing, and then feed them into MLP layers after
    concat with static attributes.
    """

    def __init__(self, inLSTM: int, hidLSTM: int, outLSTM: int, inFC: int, hidFC: Union[int, List[int]],
                 outFC: int, dropout: float = 0.5, device: Union[str, torch.device] = 'cpu'):
        """
        :param inLSTM: input size of x fed into the embedding layer before LSTM
        :param hidLSTM: hidden size of LSTM
        :param outLSTM: output size of the FC layer following LSTM
        :param inFC: input size of MLPs which equals to the sum of out_LSTM and number of static attributes
        :param hidFC: hidden size of MLPs
        :param outFC: the final output size of MLPs
        :param dropout: the dropout of fc layers
        """
        super(StaParasLSTMMLP, self).__init__()
        self.device = device
        self.hidLSTM = hidLSTM
        if isinstance(hidFC, int):
            self.hidFC = [hidFC]
        else:
            self.hidFC = hidFC

        self.fcIn = nn.Sequential(
            nn.Linear(inLSTM, hidLSTM),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.LSTM = LSTM(hidLSTM, hidLSTM, device=device)
        self.fcOut = nn.Sequential(
            nn.Linear(hidLSTM, outLSTM),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        mlpLayers = []
        inFeatures = inFC
        for i in range(len(self.hidFC)):
            outFeatures = self.hidFC[i]
            mlpLayers.append(nn.Linear(inFeatures, outFeatures))
            mlpLayers.append(nn.ReLU())
            mlpLayers.append(nn.Dropout(dropout))
            inFeatures = outFeatures
        mlpLayers.append(nn.Linear(inFeatures, outFC))
        self.MLPs = nn.Sequential(*mlpLayers)

    def forward(self, x, hidden=None):
        """
        :param x: consists of forcing and static attributes, where attributes keep the same along dimension L,
            with a shape of (N, L, F).
        """
        # feed forcing into lstm to extract features
        forcing = x[:, :, :3]
        x0 = self.fcIn(forcing)
        hidden = self.LSTM.init_hidden(x.shape[0]) if hidden is None else hidden
        x0, _ = self.LSTM(x0, hidden)
        x0 = self.fcOut(x0)

        # concat the extracted features with static attributes and feed them into MLPs
        x1 = torch.cat((x0[:, -1, :], x[:, -1, 4:]), dim=-1)
        out = self.MLPs(x1)
        return out


class StaParasConv1DMLP(nn.Module):
    def __init__(self, nx: int, nz: int, ny: int, inLen: int, hidden: int, nKernel: List[int], kernelSz: List[int],
                 stride: Union[List[int], None] = None, poolKerSz: Union[List[int], None] = None, dropout: float = 0.5):
        super(StaParasConv1DMLP, self).__init__()
        nLayer = len(nKernel)
        self.convLayers = nn.Sequential()
        inChan = nz  # need to modify the hardcode: 4 for smap and 1 for FDC
        outLen = inLen
        for i in range(nLayer):
            convLayer = CNN1dKernel(nInChannel=inChan, nKernel=nKernel[i], kernelSz=kernelSz[i], stride=stride[i])
            self.convLayers.add_module('CnnLayer%d' % (i + 1), convLayer)
            self.convLayers.add_module('Relu%d' % (i + 1), nn.ReLU())
            self.convLayers.add_module('dropout%d' % (i + 1), nn.Dropout(p=dropout))
            inChan = nKernel[i]
            outLen = calConvSize(lin=outLen, kernel=kernelSz[i], stride=stride[i])
            if poolKerSz[i] is not None:
                self.convLayers.add_module('Pooling%d' % (i + 1), nn.MaxPool1d(poolKerSz[i]))
                outLen = calPoolSize(inLen=outLen, kernelSz=poolKerSz[i])
        self.nOut = int(outLen * nKernel[-1])  # total CNN feature number after convolution
        print()
        inFC = self.nOut + nx
        self.fc = nn.Sequential(
            nn.Linear(inFC, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, ny)
        )

    def forward(self, x):
        forcing, attrs = x[:, :, :3], x[:, :, 3:]
        encoded = self.convLayers(forcing.permute(0, 2, 1))
        out = self.fc(torch.cat((encoded.squeeze(1), attrs[:, -1, :]), dim=-1))
        return out


class StaParasLSTM(nn.Module):
    """
    Use LSTM to learn the static parameters from forcing and attributes.
    """

    def __init__(self, inLSTM: int, hidLSTM: int, outLSTM: int, dropout: float = 0.5,
                 device: Union[str, torch.device] = 'cpu'):
        super(StaParasLSTM, self).__init__()
        self.device = device
        self.hidLSTM = hidLSTM
        self.fcIn = nn.Sequential(
            nn.Linear(inLSTM, hidLSTM),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.LSTM = LSTM(hidLSTM, hidLSTM, device=device)
        self.fcOut = nn.Sequential(
            nn.Linear(hidLSTM, outLSTM)
        )

    def forward(self, x, hidden=None):
        x = torch.cat([x[:, :, :3], x[:, :, 4:]], dim=-1)
        x = self.fcIn(x)
        hidden = self.LSTM.init_hidden(x.shape[0]) if hidden is None else hidden
        x, _ = self.LSTM(x, hidden)
        out = self.fcOut(x)
        return out[:, -1, :]


class RoutMLP(nn.Module):
    """
    Use FC layers to learn the parameters (K and C) of Muskingum River Routing.
    """

    def __init__(self, inSize: int, hidSize: Union[int, List[int]], outSize: int, dropout: float = 0.5,
                 device: Union[str, torch.device] = 'cpu'):
        super(RoutMLP, self).__init__()
        if isinstance(hidSize, int):
            hidSize = [hidSize]
        layers = []
        inFeatures = inSize
        for i in range(len(hidSize)):
            outFeatures = hidSize[i]
            layers.append(nn.Linear(inFeatures, outFeatures))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            inFeatures = outFeatures
        layers.append(nn.Linear(inFeatures, outSize))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: river parameters with a shape of [N, F].
        """
        out = self.net(x)
        return out


class LSTM(nn.Module):
    def __init__(self, inLSTM: int, hidLSTM: int, device: Union[str, torch.device] = 'cpu'):
        super(LSTM, self).__init__()
        self.hidLSTM = hidLSTM
        self.device = device
        self.lstm = nn.LSTM(inLSTM, hidLSTM, device=device, batch_first=True)

    def forward(self, x, hidden=None):
        hidden = self.init_hidden(x.shape[0]) if hidden is None else hidden
        out = self.lstm(x, hidden)
        return out

    def init_hidden(self, bsz):
        # LSTM h and c
        h = torch.zeros((1, bsz, self.hidLSTM), dtype=torch.float32).to(self.device)
        c = torch.zeros((1, bsz, self.hidLSTM), dtype=torch.float32).to(self.device)
        return h, c


class ExpHYDROCell(nn.Module):
    def __init__(self, routOrder: np.ndarray, area: np.ndarray, upIdx: List[int], mainIdx: int = 0,
                 pad: int = 9999, device: Union[str, torch.device] = 'cpu'):
        super(ExpHYDROCell, self).__init__()
        self.pad = pad
        self.device = device
        self.area = torch.Tensor(area).to(device)
        self.upIdx = upIdx
        self.routOrder = routOrder
        self.mainIdx = mainIdx
        self.eps = 3e-4

    @staticmethod
    def snowBucket(Sw, P, T, parTMIN, parDDF, parTMAX):
        """
        :param Sw: Initial state of snow bucket
        :param P: Precipitation
        :param T: Air temperature
        :param parTMIN: Temperature below which precipitation is snow| Range: (-3.0, 0)
        :param parDDF: Thermal degree‐day factor                     | Range: (0, 5.0)
        :param parTMAX: Temperature above which snow starts melting  | Range: (0, 3.0)
        """
        # partition rain and snow
        pr = torch.mul(P, (T >= parTMIN))
        ps = torch.mul(P, (T < parTMIN))
        melt = torch.clamp(parDDF * (T - parTMAX), min=torch.zeros_like(Sw), max=Sw)
        Sw = Sw - melt + ps

        return Sw, pr, melt

    def soilBucket(self, PET, Ssl, Sss, pr, melt, parSMAX, parF, parQMAX, parALPHA, parBETA):
        """
        :param PET: Potential evapotranspiration.
        :param Ssl: Liquid water in the soil bucket.
        :param Sss: Solid water in the soil bucket.
        :param pr: Rain.
        :param melt: Melted snow.
        :param parSMAX: Maximum storage of the catchment bucket      | Range: (100, 1500)
        :param parF: Rate of decline in flow from catchment bucket | Range: (0, 0.1)
        :param parQMAX: Maximum subsurface flow at full bucket     | Range: (10, 50)
        :param parALPHA: A parameter determining how much soil water is freezing.
        :param parBETA: a dynamic parameter used to mimic the error of PET and impacts of vegetation.
        """
        Ss = torch.clamp(Sss + Ssl, max=parSMAX - self.eps)
        freezeFlag = torch.mean(self.antT, dim=-1) < 0  # soil is freezing when flag is true and thawing otherwise.
        minSss = torch.where(freezeFlag, Sss, torch.zeros_like(Sss))
        # the solid water will decrease when thawing, so the Sus at timestep t should be no more than that at t-1
        # subtract a small number to make sure Sus is smaller than Su, so that Sul will not be 0 for gradient tracking
        maxSss = torch.where(freezeFlag, Ss, Sss)
        Sss = minSss + (maxSss - minSss) * parALPHA
        Ssl = torch.clamp(Ss - Sss, min=self.eps)

        Qb = torch.clamp(parQMAX * torch.exp(-1 * parF * (parSMAX - Sss - Ssl)), max=torch.max(Ssl-self.eps, parQMAX),
                         min=torch.zeros_like(parQMAX))

        cr = torch.clamp((Ssl / (parSMAX - Sss)) ** parBETA, max=1, min=0)
        et = torch.clamp(PET * cr, max=Ssl - Qb)

        Ssl = Ssl + pr + melt - et - Qb
        Qs = torch.clamp(Ssl - (parSMAX - Sss), min=0)
        Ssl = Ssl - Qs

        return Ssl, Sss, et, Qb, Qs

    def hillSlopeRouting(self, Q, parA, parB, tmax=15):
        """
        Gamma distribution for routing:
                γ(t:a,b) = 1 / (Γ(a) * b^a) * t^(a-1) * e^(-t/b)
                q(t) = ∫(_0^tmax)((γ(t:a,b) * R(t-s))ds

        Reference
        ---------
        Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable, regionalized process‐based
        models with multiphysical outputs can approach state‐of‐the‐art hydrologic prediction accuracy. Water
        Resources Research. https://doi.org/10.1029/2022wr032404

        :param Q: simulated streamflow output by lumped model with a shape of (N, L)
        :param parA: shape parameter for gamma distribution, with a shape of (N, L)
        :param parB: timescale parameter for gamma distribution, with a shape of (N, L)
        :param tmax: maximum time length for unit hydrograph, int
        """

        def UH_conv(x, UH):
            """
            :param x: streamflow output by lumped models, with a shape of [N, F, L]
            :param UH: unit hydrograph, with a shape of [N, F, tmax]
            """
            basin_size, channel_size, sequence_length = x.shape
            kernel_size = UH.shape[-1]

            # batch and basins need to do convolution dependently and we make use of groups
            groups = basin_size
            xx = x.view([channel_size, groups, sequence_length])
            w = UH.view([groups, channel_size, kernel_size])

            # Q(t)=\integral(x(\tao)*UH(t-\tao))d\tao, conv1d does \integral(w(\tao)*x(t+\tao))d\tao, hence we flip UH
            y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=kernel_size - 1, stride=1, bias=None)
            y = y[:, :, 0:-(kernel_size - 1)]
            return y.view(x.shape)

        def UH_gamma(a, b, tmax):
            """
            :param a: shape parameter, range(0, 2.9), with a shape of (L, N, 1)
            :param b: timescale parameter, range(0, 6.5), with a shape of (L, N, 1)
            :param tmax: maximum time length for gamma distribution

            :return
                w: weights of the unit graph for different timestep, with a shape of (tmax, N, 1)
            """
            m = a.shape
            w = torch.zeros([tmax, m[1], m[2]])
            aa = F.relu(a[0:tmax, :, :]).view([tmax, m[1], m[2]]) + 0.1  # minimum 0.1. First dimension of a is repeat
            theta = F.relu(b[0:tmax, :, :]).view([tmax, m[1], m[2]]) + 0.5  # minimum 0.5
            t = torch.arange(0.5, tmax * 1.0).view([tmax, 1, 1]).repeat([1, m[1], m[2]])
            t = t.to(aa.device)
            denom = (aa.lgamma().exp()) * (theta ** aa)
            mid = t ** (aa - 1)
            right = torch.exp(-t / theta)
            w = 1 / denom * mid * right
            w = w / w.sum(0)  # scale to 1 for each UH

            return w

        routA = parA.unsqueeze(-1).permute(1, 0, 2)  # (N, L) to # (L, N, 1)
        routB = parB.unsqueeze(-1).permute(1, 0, 2)
        UH = UH_gamma(routA, routB, tmax)
        UH = UH.permute(1, 2, 0)  # (tmax, N, 1) to (N, 1, tmax)
        Qin = Q.unsqueeze(1)  # (N, L) to (N, 1, L)
        Q = UH_conv(Qin, UH)

        return Q.squeeze(1)

    def init_model(self, R0=None, S0=None, SD0=None, tLen=5):
        """
        :param R0: initial inflow and outflow of each river reach.
        :param S0: initial reservoir storage of Flex model.
        :param SD0: initial state fed into lstm cell if LSTMCell is selected to predict dynamic parameters.
        :param tLen: used to initiate a tensor with the length of 'tLen' to store the past 'tLen' days of temperature.
            Soil is freezing when the average mean of the tensor is below 0 and thawing when the mean is above 0.
        """
        # divide the routing order array into mainstream and tributary sub-basins
        self.main = self.routOrder[self.mainIdx]
        self.trib = self.routOrder[[i for i in range(len(self.routOrder)) if i != self.mainIdx]]
        # count the number of non-pad values in tempTrib
        nb = np.count_nonzero(self.trib != self.pad, axis=1)
        self.trib = self.trib[:, :np.max(nb) + 1]

        # initiate inflow and outflow of each river reach
        nrivers = len(self.upIdx)  # each reach corresponds to an upstream sub-basin
        if R0 is None:
            R0 = (torch.zeros((nrivers, 2), dtype=torch.float32) + 0.001).to(self.device)

        nbasin = nrivers + 1  # each sub-basin corresponds a reach to routing except for the most downstream one
        # initiate five reservoir storage
        if S0 is None:
            S0 = (torch.zeros([nbasin, 3], dtype=torch.float32) + 0.001).to(self.device)

        # initiate frozen water storage which will be fed into lstm cell to get dynamic parameters
        if SD0 is None:
            Sw, Ssl0, Sss0 = S0[:, 0], S0[:, 1], S0[:, 2]
            SD0 = torch.stack((Ssl0, Sss0), dim=1)

        # initiate a tensor to storage antecedent temperature
        self.antT = torch.zeros([nbasin, tLen], dtype=torch.float32).to(self.device)

        return R0, S0, SD0

    def channelRouting(self, Qh, R0, parX, parK):
        """
        :param Qh: simulated hillslope-routed flow (mm/d) of sub-basins with a shape of [N_basin, L].
        :param parX: one of muskingum parameters ranges (0, 0.5) with a shape of [N_river, L].
        :param parK: anothe rmuskingum parameter satisfying 2𝐾𝑋<Δt≤𝐾 with a shape of [N_river, L].
        :param R0: the initial inflow and outflow of river reaches with a shape of [N_river, 2].

        :return: a tuple containing two elements
            Qr: the final runoff at the outlet of each sub-basin.
        """

        # calculate C1, C2, C3 in the muskingum method
        dT = 1  # simulation time step: 1 day
        C1 = (dT - 2 * parK * parX) / (2 * parK * (1 - parX) + dT)
        C2 = (dT + 2 * parK * parX) / (2 * parK * (1 - parX) + dT)
        C3 = (2 * parK * (1 - parX) - dT) / (2 * parK * (1 - parX) + dT)

        # initiate tensors to store the final runoff of sub-basins
        Qh = Qh * self.area.unsqueeze(1).repeat(1, Qh.size(1)) * 1000 / (3600 * 24)  # convert qh from mm/d to m3/s
        Qr = torch.zeros(Qh.size(), dtype=torch.float32).to(self.device)
        # initiate tensors to store the inflow and outflow of reaches
        inFlow0, outFlow0 = R0[:, 0].clone(), R0[:, 1].clone()
        inFlowt = torch.zeros(inFlow0.size(), dtype=torch.float32).to(self.device)
        outFlowt = torch.zeros(outFlow0.size(), dtype=torch.float32).to(self.device)

        for t in range(Qh.size(1)):
            # calculate the channel routing of tributary sub-basins first
            for i in range(self.trib.shape[1] - 1):
                bsns = np.array(
                    [self.trib[j, i] for j in range(self.trib.shape[0]) if (self.trib[j, i + 1] != self.pad)])
                # determine the index of river reach corresponding the sub-basins
                idx = [self.upIdx.index(bsn) for bsn in bsns]
                # if the sub-basin is at the most upstream, only the hillslope-routed flow needs to be considered.
                if i == 0:
                    inFlowt[idx] = Qh[bsns, t]
                # otherwise, the channel-routed flow from the outlet fo last reach also needs to be considered.
                else:
                    # determine the last river reach by the nearest upstream sub-basin
                    upBsns = np.array(
                        [self.trib[j, i - 1] for j in range(self.trib.shape[0]) if (self.trib[j, i + 1] != self.pad)])
                    inFlowt[idx] = Qh[bsns, t] + outFlowt[[self.upIdx.index(upBsn) for upBsn in upBsns]]
                outFlowt[idx] = C1[idx, t] * inFlowt[idx] + C2[idx, t] * inFlow0[idx] + C3[idx, t] * outFlow0[idx]

            # calculate the flow routine of mainstream sub-basins then
            mainbsns = np.array([self.main[i] for i in range(len(self.main) - 1) if (self.main[i + 1] != self.pad)])
            for i, bsn in enumerate(mainbsns):
                idx = self.upIdx.index(bsn)
                # if the sub-basin is at the most upstream, only the hillslope-routed flow needs to be considered.
                if i == 0:
                    inFlowt[idx] = Qh[bsn, t]
                # otherwise, the channel-routed flow from the outlet fo last reach also needs to be considered.
                else:
                    # determine the last river reach by the nearest upstream sub-basin, including the mainstream and
                    # tributary sub-basins
                    rowIdx, colIdx = np.where(self.routOrder == bsn)
                    upBsns = self.routOrder[rowIdx, colIdx - 1]
                    inFlowt[idx] = Qh[bsn, t] + outFlowt[[self.upIdx.index(upBsn) for upBsn in upBsns]].sum()
                outFlowt[idx] = C1[idx, t] * inFlowt[idx].clone() + C2[idx, t] * inFlow0[idx] + C3[idx, t] * outFlow0[
                    idx]

            # calculate the runoff at the outlet of each sub-basin at the current timestep t
            # the runoff at the outlet of each sub-basin is the inflow of corresponding reach except for the most downstream one
            Qr[self.upIdx, t] = inFlowt
            # find out the most downstream sub-basin and its nearest upstream sub-basins
            outBsn = [i for i in range(len(Qr)) if i not in self.upIdx]
            rowIdx, colIdx = np.where(self.routOrder == outBsn)
            upBsns = self.routOrder[rowIdx, colIdx - 1]
            Qr[outBsn, t] = Qh[outBsn, t] + outFlowt[upBsns].sum() if len(upBsns) > 1 else Qh[outBsn, t] + outFlowt[
                upBsns]
            inFlow0, outFlow0 = inFlowt.clone(), outFlowt.clone()
        return Qr

    def forward(self, forcing, S0, staParas, dynParas):
        """
        :param forcing: tensor of shape[N, F), ie., [number of basins, feature size], consisting of
            precipitation, temperature, and potential evapotranspiration at the current timestep t.
        :param S0: tensor of shape [N, 3], consisting of the initial storage of snow, liquid soil water, solid soil
            water storage at the last timestep t-1.
        :param staParas: tensor of shape [N, 6], including parTT, parDDF, parQMAX, parA, and parB,
            parSMAX at the current timestep t.
        :param dynParas: tensor of shape [N, 2], including parF and parALPHA at the current timestep.
        :return
            output: a dict consisting final runoff, reservoir and river storage, as well as other hydrological
            variables at current timestep t.
        """

        # forcing
        P, T, Ep = forcing[:, 0], forcing[:, 1], forcing[:, 2]
        self.antT = torch.cat((self.antT[:, -(len(self.antT) - 1):].clone(), T.unsqueeze(-1)), dim=-1)

        # initiate the storages
        Sw0, Ssl0, Sss0 = S0[:, 0], S0[:, 1], S0[:, 2]

        # static parameters
        # parTT, parDDF, parQMAX, parSMAX = staParas[:, 0], staParas[:, 1], staParas[:, 2], staParas[:, 5]
        parTMIN, parDDF, parTMAX, parQMAX = staParas[:, 0], staParas[:, 1], staParas[:, 2], staParas[:, 3]
        parSMAX = staParas[:, 6]

        # dynamic parameters
        parF, parALPHA, parBETA = dynParas[:, 0], dynParas[:, 1], dynParas[:, 2]

        # snow bucket
        Swt, Pr, M = self.snowBucket(Sw0, P, T, parTMIN, parDDF, parTMAX)

        # soil bucket
        Sslt, Ssst, E, Qb, Qs = self.soilBucket(Ep, Ssl0, Sss0, Pr, M, parSMAX, parF, parQMAX, parALPHA, parBETA)
        Sst = Sslt + Ssst

        output = {'Swt': Swt, 'Sst': Sst,  # final reservoir storage
                  'Qb': Qb, 'Qs': Qs,  # subsurface flow and surface flow
                  'E': E,  # three evapotranspiration
                  'Sslt': Sslt, 'Ssst': Ssst, 'alpha': parALPHA}  # liquid and solid water in soil bucket

        return output



class CNN1dKernel(torch.nn.Module):
    def __init__(self, *, nInChannel=1, nKernel=3, kernelSz=3, stride=1, padding=0):
        super(CNN1dKernel, self).__init__()
        self.cnn1d = nn.Conv1d(in_channels=nInChannel, out_channels=nKernel, kernel_size=kernelSz, padding=padding,
                               stride=stride)

    def forward(self, x):
        output = F.relu(self.cnn1d(x))
        return output


def calConvSize(lin, kernel, stride, padding=0, dilation=1):
    lout = (lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return int(lout)


def calPoolSize(inLen, kernelSz, stride=None, padding=0, dilation=1):
    if stride is None:
        stride = kernelSz
    outLen = (inLen + 2 * padding - dilation * (kernelSz - 1) - 1) / stride + 1
    return int(outLen)

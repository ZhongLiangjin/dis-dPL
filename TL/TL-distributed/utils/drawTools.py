import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Union
import geopandas as gpd
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter
import seaborn as sns


def visFlow(flowFile: str, savePath: str, tScale: str, tRange: Union[None, List[str]] = None):
    """ Used to visualize observed streamflow data for quality control. """
    flowTemp = pd.read_csv(flowFile, index_col='DATE', parse_dates=True)
    # convert units from m^3 to mm/d
    area = {'HHY': 20930, 'JIM': 45019, 'JUG': 98414, 'MAQ': 86048, 'MET': 59655, 'TNH': 121972}
    for station in list(flowTemp.columns):
        flowTemp[station] = flowTemp[station] * 24 * 3600 / (area[station] * 1000)
    # load data in given period and reverse the columns
    if tRange is not None:
        tRange = pd.date_range(start=tRange[0], end=tRange[1])
    else:
        tRange = flowTemp.index
    flowTemp = flowTemp[flowTemp.index.isin(tRange)].loc[:, ::-1]
    # resample to the given timescale
    if tScale == 'd':
        flow = flowTemp.values
        tPlot = flowTemp.index
    elif tScale == 'm':
        tPlot = pd.Series(data=flowTemp.index.values, index=flowTemp.index).resample('M').first()
        flow = np.zeros((len(tPlot), len(flowTemp.columns)))
        for i, station in enumerate(list(flowTemp.columns)):
            flow[:, i] = flowTemp[station].groupby([flowTemp.index.year, flowTemp.index.month]).apply(
                lambda x: np.nansum(x.values)).values
        flow[flow == 0] = np.nan
    else:
        raise ValueError('tScale must be either d or m')

    # create a figure to plot
    fig, axes = plt.subplots(figsize=(10, 10), nrows=6, ncols=1)
    for i, station in enumerate(list(flowTemp.columns)):
        ax = axes[i]
        ax.plot(tPlot.values, flow[:, i])
        ax.set_ylabel(f'$Q (mm/{tScale})$')
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
        ax.set_xlim(tPlot[0] - pd.offsets.Day(60), tPlot[-1] + pd.offsets.Day(60))
        ax.tick_params(axis='x', labelrotation=30)
        ax.annotate(f'{chr(97 + i)}. {station}', xy=(0.01, 0.85), xycoords='axes fraction')
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig(savePath)
    plt.show()


def checkMet(dataFile: str, tRange: List[str], basinFile: str, var: str, figSize: tuple = (10, 2.5),
             cmap: str = 'viridis_r', vMinMax: Union[List, None] = None, savePath: Union[None, str] = None):
    import warnings
    warnings.filterwarnings('ignore')  # ignore the warning of plt.tight_layout()
    plt.rcParams['font.family'] = 'Arial'
    sns.set(style='ticks')

    basins = gpd.read_file(basinFile)
    tRange = pd.date_range(tRange[0], tRange[1])
    with open(dataFile, 'rb') as f:
        data = pickle.load(f)

    unitDict = dict(zip(['tas', 'pr', 'pet'], [r'($^\circ$C)', r'(mm/d)', r'(mm/d)']))
    assert var in unitDict.keys()

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    dfMet = pd.DataFrame(index=range(0, len(basins)), columns=months)
    for i in range(len(basins)):
        metTemp = data['basin_forcing_attr'][f'basin_{i}']['forcing'][var]
        metTemp = metTemp[metTemp.index.isin(tRange)]
        dfMet.loc[i, :] = metTemp.groupby(metTemp.index.month).mean().T.values
    dfPlot = pd.concat([basins, dfMet], axis=1)

    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=figSize, dpi=300)
    cbarAx = fig.add_axes([0.907, 0.17, 0.007, 0.656])  # left, bottom, width, height
    if vMinMax is None:
        vMin, vMax = dfPlot[months].min().min(), dfPlot[months].max().max()
    else:
        vMin, vMax = vMinMax
    for i, month in enumerate(months):
        axRow, axCol = int(i / 6), i % 6
        ax = axes[axRow, axCol]
        dfPlot.plot(column=month, ax=ax, cmap=cmap, edgecolor='None', linewidth=0.1, alpha=0.8)
        ax.annotate(month, xy=(0.03, 0.05), xycoords='axes fraction', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        lw = 0.8
        ax.spines['top'].set_linewidth(lw)  # Top spine thickness
        ax.spines['bottom'].set_linewidth(lw)  # Bottom spine thickness
        ax.spines['left'].set_linewidth(lw)  # Left spine thickness
        ax.spines['right'].set_linewidth(lw)  # Right spine thickness

    norm = colors.Normalize(vmin=vMin, vmax=vMax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, cax=cbarAx, orientation='vertical', alpha=0.8)
    cbar.ax.set_ylabel(f'{var} {unitDict[var]}', fontsize=10)
    cbar.ax.tick_params(direction='out', length=2, labelsize=10)
    cbar.outline.set_linewidth(lw)

    plt.subplots_adjust(hspace=-0.15, wspace=0.1)
    if savePath is not None:
        plt.savefig(savePath)
    plt.show()


def checkAttrs(dataFile: str, basinFile: str, figSize: tuple = (12, 8), cmap: str = 'viridis_r'):
    import warnings
    warnings.filterwarnings('ignore')  # ignore the warning of plt.tight_layout()
    plt.rcParams['font.family'] = 'Arial'
    sns.set(style='ticks')

    basins = gpd.read_file(basinFile)
    with open(dataFile, "rb") as f:
        data = pickle.load(f)

    excludeAttrs = ['elev_min', 'elev_max', 'elev_std', 'slope_max', 'slope_min', 'slope_std', 'flowlen_min',
                    'flowlen_max', 'flowlen_std', 'sand_min', 'sand_max', 'sand_std', 'clay_min', 'clay_max',
                    'clay_std', 'silt_min', 'silt_max', 'silt_std', 'porosity_min', 'porosity_max', 'porosity_std',
                    'PDep_min', 'PDep_max', 'PDep_std', 'DTB_min', 'DTB_max', 'DTB_std', 'theta_s_min', 'theta_s_max',
                    'theta_s_std', 'log_k_s_min', 'log_k_s_max', 'log_k_s_std', 'tksatu_min', 'tksatu_max',
                    'tksatu_std', 'tksatf_min', 'tksatf_max', 'tksatf_std', 'tkdry_min', 'tkdry_max', 'tkdry_std']

    attrs = [a for a in data['basin_forcing_attr']['basin_0']['attrs'].keys() if a not in excludeAttrs]
    nameAttrDict = dict()
    for attr in attrs:
        nameAttrDict[attr] = attr[:-5] if attr.endswith('_mean') and attr.startswith('lai') is False else attr

    dfAttr = pd.DataFrame(index=range(0, len(basins)), columns=attrs)
    for i in range(len(basins)):
        for attr in attrs:
            dfAttr.loc[i, attr] = data['basin_forcing_attr'][f'basin_{i}']['attrs'][attr]
    dfPlot = pd.concat([basins, dfAttr], axis=1)

    fig, axes = plt.subplots(nrows=6, ncols=5, figsize=figSize, dpi=300)
    for i, attr in enumerate(attrs):
        vMin, vMax = dfPlot[attr].min(), dfPlot[attr].max()
        axRow, axCol = int(i / 5), i % 5
        ax = axes[axRow, axCol]
        dfPlot.plot(column=attr, ax=ax, cmap=cmap, edgecolor='None', linewidth=0.1, alpha=0.8)
        ax.annotate(nameAttrDict[attr], xy=(0.05, 0.08), xycoords='axes fraction')
        ax.set_xticks([])
        ax.set_yticks([])

        lw = 0.8
        ax.spines['top'].set_linewidth(lw)  # Top spine thickness
        ax.spines['bottom'].set_linewidth(lw)  # Bottom spine thickness
        ax.spines['left'].set_linewidth(lw)  # Left spine thickness
        ax.spines['right'].set_linewidth(lw)  # Right spine thickness

        norm = colors.Normalize(vmin=vMin, vmax=vMax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbFmt = ScalarFormatterForceFormat()  # create the formatter
        cbFmt.set_powerlimits((-1, 1))
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', format=cbFmt, shrink=1, alpha=0.8)
        cbar.ax.tick_params(direction='out', length=2, labelsize=10)
        cbar.outline.set_linewidth(lw)
    plt.delaxes(axes[-1, -1])

    plt.tight_layout()  # left, bottom, right, top
    plt.savefig('attributes.pdf')
    plt.show()


def checkParTT(dir: str, basinFile: str, figSize: tuple = (6, 4), cmap: str = 'viridis_r'):
    for folder in os.listdir(dir):
        path = os.path.join(dir, folder, 'TT.npy')
        if os.path.exists(path):
            TT = np.load(path)
            basins = gpd.read_file(basinFile)
            dfPlot = basins.copy()
            dfPlot['TT'] = TT

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figSize, dpi=300)
            vMin, vMax = dfPlot['TT'].min(), dfPlot['TT'].max()
            dfPlot.plot(column='TT', ax=ax, cmap=cmap)
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            ax.yaxis.set_major_locator(plt.MultipleLocator(1))

            norm = colors.Normalize(vmin=vMin, vmax=vMax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.75)
            cbar.ax.set_ylabel(r'TT ($^\circ$C)')

            plt.tight_layout()
            plt.savefig(os.path.join(dir, folder, 'TT.png'))
            plt.show()

def drawBsn(basinFile='../../Data/GIS/watershed_level1.shp', streamFile='../../Data/GIS/river_routine_WGS84.shp'):
    plt.rcParams['font.family'] = 'Arial'
    sns.set(style="ticks")
    from matplotlib.lines import Line2D

    river = gpd.read_file(streamFile)
    basin = gpd.read_file(basinFile)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8), dpi=300)
    basin.plot(ax=ax, color=(0.90625, 0.91796875, 0.93359375), edgecolor=(0.5, 0.5, 0.5, 0.7), linewidth=0.5)
    xText = [poly.centroid.x for poly in basin.geometry]
    yText = [poly.centroid.y for poly in basin.geometry]
    for x, y, label in zip(xText, yText, basin.loc[:, 'level']):
        ax.annotate(label, xy=(x, y), xytext=(-8, -5), textcoords="offset points")
    river.plot(ax=ax, color=(0.0390625, 0.57421875, 0.984375), linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles = []
    handles.append(ax.bar([], [], color=(0.90625, 0.91796875, 0.93359375), edgecolor=(0.5, 0.5, 0.5, 0.7), label='sub-basins'))
    handles.append(ax.plot([], [], color=(0.0390625, 0.57421875, 0.984375), linewidth=1, label='river reach')[0])
    fig.legend(handles=handles, bbox_to_anchor=(0.03, 0.03, 0.2, 0.02), frameon=False, ncol=2,
               columnspacing=1, borderaxespad=0.1, handletextpad=0.1)

    plt.savefig('basin.pdf')
    fig.show()


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here


if __name__ == '__main__':
    # visFlow(flowFile='../data/streamflow.csv', savePath='flow_M.png', tScale='m')
    # visFlow(flowFile='../data/streamflow.csv', savePath='flow_D.png', tScale='d', tRange=['1990-01-01', '2019-12-31'])

    # visualize meteorological data
    # checkMet(dataFile='../data/data.pkl', basinFile='../data/sub-basins/watershed.shp', cmap='coolwarm', var='tas',
    #          tRange=['2000-01-01', '2019-12-31'], savePath='tas.pdf')
    # checkMet(dataFile='../data/data.pkl', basinFile='../data/sub-basins/watershed.shp', cmap='coolwarm_r', var='pr',
    #          tRange=['2000-01-01', '2019-12-31'],  savePath='pr.pdf', vMinMax=[0.064, 6.488])
    # checkMet(dataFile='../data/data.pkl', basinFile='../data/sub-basins/watershed.shp', cmap='coolwarm', var='pet',
    #          tRange=['2000-01-01', '2019-12-31'], savePath='pet.pdf')

    # visualize parTT
    # checkParTT(dir='../checkpoints/EXP-HYDRO', basinFile='../data/sub-basins/watershed.shp', cmap='coolwarm')

    # visualize basin attributes
    checkAttrs(dataFile='../data/data.pkl', basinFile='../data/sub-basins/watershed.shp', cmap='coolwarm')
    # drawBsn()



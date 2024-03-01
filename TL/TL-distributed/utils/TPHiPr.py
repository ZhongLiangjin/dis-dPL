import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Tuple, Union
import os
import netCDF4 as nc
from matplotlib import pyplot as plt, colors
from rasterio.transform import Affine
import rasterio
import rasterio.mask
import fiona
import seaborn as sns


class VisTPHiPr:
    def __init__(self, basinFile: str, tRange: List[str], TPHiPrFile: Union[str, None] = None,
                 dataDir: Union[str, None] = None):

        self.tRange = pd.date_range(tRange[0], tRange[1], freq='m')
        self.basins = gpd.read_file(basinFile)
        with fiona.open(basinFile, 'r') as shapefile:
            self.shapes = [feature['geometry'] for feature in shapefile]
        if TPHiPrFile is None:
            self.dataDir = dataDir
            self.TPHiPr = self.loadData()
        else:
            self.TPHiPr = pd.read_csv(TPHiPrFile, parse_dates=True, index_col='date')

    def loadData(self, var: str = 'prcp', outPath: str = '../data/TPHiPr.csv'):
        # create a dataframe to store data
        dfPr = pd.DataFrame(index=self.tRange, columns=[f'basin_{i}' for i in range(0, len(self.basins))])
        dfPr.index.name = 'Date'

        # time loop to read .nc file
        for t in self.tRange:
            date = t.strftime('%Y%m%d')
            f = nc.Dataset(os.path.join(self.dataDir, f'tpmfd_prcp_m_{date[:6]}.nc'))
            print(f'Processing tpmfd_prcp_m_{date[:6]}.nc')
            lat, lon = np.array(f['latitude']), np.array(f['longitude'])
            dataset, transform = self.cropByRoi(dataset=np.array(f[var][:]), x0=lon[0], y0=lat[0],
                                                res_x=lon[1] - lon[0], res_y=lat[1] - lat[0],
                                                roi=[95.89, 32.16, 103.41, 36.12])
            dataset = dataset.squeeze(0) * 24
            dataset[(dataset < -999) | (dataset > 100)] = np.nan
            f.close()

            # store data into a Tiff file to mask later
            with rasterio.open(
                    'tmp.tif',
                    'w',
                    driver='GTiff',
                    height=dataset.shape[0],
                    width=dataset.shape[1],
                    count=1,
                    dtype=dataset.dtype,
                    crs='+proj=latlong',
                    transform=transform,
                    nodata=np.nan,
            ) as dst:
                dst.write(dataset, 1)

            # mask the Tiff file by sub-basins
            with rasterio.open('tmp.tif') as src:
                for i, shape in enumerate(self.shapes):
                    out_image, _ = rasterio.mask.mask(src, [shape], crop=True)
                    out_image = out_image.astype('float64') if out_image.dtype != 'float64' else out_image
                    dfPr.loc[t, f'basin_{i}'] = np.nanmean(out_image)
            os.remove('tmp.tif')

        dfPr.to_csv(outPath)
        return dfPr

    def plotPr(self, figSize: tuple = (10, 2.5), cmap: str = 'viridis_r'):
        import warnings
        warnings.filterwarnings('ignore')  # ignore the warning of plt.tight_layout()
        plt.rcParams['font.family'] = 'Arial'
        sns.set(style='ticks')

        # get annual mean precipitation for 12 months
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        prTemp = self.TPHiPr.groupby(self.TPHiPr.index.month).mean().T
        prTemp.columns, prTemp.index = months, range(0, 69)
        dfPlot = pd.concat([self.basins, prTemp], axis=1)

        # plot pr
        fig, axes = plt.subplots(nrows=2, ncols=6, figsize=figSize, dpi=300)
        cbarAx = fig.add_axes([0.907, 0.17, 0.007, 0.656])  # left, bottom, width, height
        vMin, vMax = dfPlot[months].min().min(), dfPlot[months].max().max()
        print(f'vMin: {vMin: .3f}, vMax: {vMax: .3f}')
        for i, month in enumerate(months):
            axRow, axCol = int(i / 6), i % 6
            ax = axes[axRow, axCol]
            dfPlot.plot(column=month, ax=ax, cmap=cmap, edgecolor='None', linewidth=0.1, alpha=0.8)
            ax.annotate(month, xy=(0.03, 0.05), xycoords='axes fraction', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

            # Set axis spines properties
            lw = 0.8
            ax.spines['top'].set_linewidth(lw)  # Top spine thickness
            ax.spines['bottom'].set_linewidth(lw)  # Bottom spine thickness
            ax.spines['left'].set_linewidth(lw)  # Left spine thickness
            ax.spines['right'].set_linewidth(lw)  # Right spine thickness

        norm = colors.Normalize(vmin=vMin, vmax=vMax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, cax=cbarAx, orientation='vertical', alpha=0.8)
        cbar.ax.set_ylabel(f'TPHiPr (mm/d)', fontsize=10)

        # plt.tight_layout(rect=[0, 0, 0.945, 1])  # left, bottom, right, top
        plt.subplots_adjust(hspace=-0.15, wspace=0.1)
        plt.savefig('TPHiPr.pdf')
        plt.show()


    @staticmethod
    def cropByRoi(dataset: np.ndarray, x0: float, y0: float, res_x: float, res_y: float,
                  roi: list) -> Tuple[np.ndarray, Affine]:
        """
        Parameters
            dataset: the dataset with a much larger range, with 2 or 3 dimensions
            x0: the start longitude of the dataset
            y0: the start latitude of the dataset
            res_x: the resolution of longitude
            res_y: the resolution of latitude
            roi: the rectangular range of study area, [left, bottom, right, top], such as [95.89, 32.16, 103.41, 36.12]

        Return
             a masked dataset within the roi and an affine transformation matrix for the masked dataset
        """
        assert dataset.ndim in [2, 3]
        dataset = np.expand_dims(dataset, axis=0) if dataset.ndim == 2 else dataset
        idx_left = int(np.floor((roi[0] - x0) / res_x))
        idx_right = int(np.ceil((roi[2] - x0) / res_x))
        if res_y < 0:  # a negative res_y means the latitude ranges from top to bottom
            idx_bottom = int(np.ceil((roi[1] - y0) / res_y))
            idx_top = int(np.floor((roi[3] - y0) / res_y))
            new_dataset = dataset[:, idx_top: idx_bottom, idx_left: idx_right]
            transform = Affine.translation(x0 + idx_left * res_x + res_x / 2,
                                           y0 + idx_top * res_y + res_y / 2) * Affine.scale(res_x, res_y)
        else:
            idx_bottom = int(np.floor((roi[1] - y0) / res_y))
            idx_top = int(np.ceil((roi[3] - y0) / res_y))
            new_dataset = dataset[:, idx_bottom: idx_top, idx_left: idx_right]
            transform = Affine.translation(x0 + idx_left * res_x + res_x / 2,
                                           y0 + idx_top * res_y + res_y / 2) * Affine.scale(res_x, -res_y)

        return new_dataset, transform


if __name__ == '__main__':
    THPiPrData = VisTPHiPr(TPHiPrFile='../data/TPHiPr.csv', basinFile='../data/sub-basins/watershed.shp',
                           tRange=['2000-1-1', '2019-12-31'])
    THPiPrData.plotPr(cmap=sns.color_palette('coolwarm_r', as_cmap=True))

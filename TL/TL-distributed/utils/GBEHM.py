import pickle
from typing import Union, List
import os
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import fiona
from collections import OrderedDict

class GBEHM:
    def __init__(self, root='E:/Research/dPL/Data/GBEHM',
                 basinFile='E:/Research/dPL/Data/GIS/watershed_sryr.shp',
                 tRange=['2000-1-1', '2019-12-31'], stations=['TNH', 'JUG', 'MAQ', 'MET', 'JIM']):
        self.root = root
        with fiona.open(basinFile, 'r') as shapefile:
            self.shapes = [feature['geometry'] for feature in shapefile]
        self.Qs = self.load_runoff(stations=stations, tRange=tRange)
        self.ET = self.load_ET(tRange=tRange)
        self.snowDep = self.load_snow_depth(tRange=tRange)
        self.soiwice, self.soiwliq = self.load_soil_water(tRange=tRange)


    def load_runoff(self, stations: List[str], tRange: List[str]):
        tRange = pd.date_range(tRange[0], tRange[1])
        df = pd.DataFrame(index=tRange, columns=stations)
        code = {'TNH': 4019, 'JUG': 4011, 'MAQ': 4001, 'MET': 3035, 'JIM': 3024, 'HHY': 3012}  # code for station
        area = {'HHY': 20930, 'JIM': 45019, 'JUG': 98414, 'MAQ': 86048, 'MET': 59655, 'TNH': 121972}  # drainage area
        for station in stations:
            # read runoff data from file
            filePath = os.path.join(self.root, 'river', f'river_ws{code[station]}.daily')
            data = pd.read_csv(filePath, sep=r'\s+', header=None, names=['Year', 'Month', 'Day', 'Runoff'])
            data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
            data.drop(['Year', 'Month', 'Day'], axis=1, inplace=True)
            data.set_index('Date', inplace=True)
            df.loc[:, station] = data[data.index.isin(tRange)].values * 24 * 3600 / (area[station] * 1000)
        return df

    def load_ET(self, tRange: List[str]):
        tRange = pd.date_range(tRange[0], tRange[1], freq='m')
        df = pd.DataFrame(columns=range(0, 69), index=tRange)
        for t in df.index:
            file = os.path.join(self.root, 'spatial', 'evap', f"evap_{t.strftime('%Y%m')}.asc")
            print(f"Processing evap_{t.strftime('%Y%m')}.asc")
            with rasterio.open(file) as src:
                for j, shape in enumerate(self.shapes):
                    out_image, _ = rasterio.mask.mask(src, [shape], crop=True, all_touched=True)
                    out_image[out_image == -9999] = np.nan
                    out_image[out_image < 0] = 0
                    df.loc[t, j] = np.nanmean(out_image)
        return df

    def load_snow_depth(self, tRange: List[str]):
        tRange = pd.date_range(tRange[0], tRange[1])
        df = pd.DataFrame(columns=range(0, 69), index=tRange)
        for t in df.index:
            file = os.path.join(self.root, 'spatial', 'snowds', f"snowds_{t.strftime('%Y%m%d')}.asc")
            print(f"Processing snowds_{t.strftime('%Y%m%d')}.asc")
            with rasterio.open(file) as src:
                for j, shape in enumerate(self.shapes):
                    out_image, _ = rasterio.mask.mask(src, [shape], crop=True, all_touched=True)
                    out_image[out_image == -9999] = np.nan
                    out_image[out_image < 0] = 0
                    df.loc[t, j] = np.nanmean(out_image)
        return df

    def load_soil_water(self, tRange: List[str], layers=[1, 21]):
        tRange = pd.date_range(tRange[0], tRange[1], freq='m')
        df_soiwice = pd.DataFrame(columns=range(0, 69), index=tRange)
        df_soiwliq = pd.DataFrame(columns=range(0, 69), index=tRange)
        for t in tRange:
            print(f"Processing {t.strftime('%Y%m')}")
            soiwiceTemp = np.zeros([layers[-1]-layers[0]+1, 69], dtype=float)
            soiwliqTemp = np.zeros([layers[-1]-layers[0]+1, 69], dtype=float)
            for i in np.arange(layers[0], layers[-1]+1):
                soil_layer = str(i) if i >= 10 else f'0{i}'
                file_liq = os.path.join(self.root, 'spatial', 'soil', f"soiwliq_{t.strftime('%Y%m')}{soil_layer}.asc")
                with rasterio.open(file_liq) as src:
                    for j, shape in enumerate(self.shapes):
                        out_image, _ = rasterio.mask.mask(src, [shape], crop=True, all_touched=True)
                        out_image[out_image == -9999] = np.nan
                        soiwliqTemp[i-1, j] = np.nanmean(out_image)

                file_ice = os.path.join(self.root, 'spatial', 'soil', f"soiwice_{t.strftime('%Y%m')}{soil_layer}.asc")
                with rasterio.open(file_ice) as src:
                    for j, shape in enumerate(self.shapes):
                        out_image, _ = rasterio.mask.mask(src, [shape], crop=True, all_touched=True)
                        out_image[out_image == -9999] = np.nan
                        soiwiceTemp[i-1, j] = np.nanmean(out_image)
            df_soiwice.loc[t, :] = np.nanmean(soiwiceTemp, axis=0)
            df_soiwliq.loc[t, :] = np.nanmean(soiwliqTemp, axis=0)

        return df_soiwice, df_soiwliq



if __name__ == '__main__':
    model = GBEHM(root='E:/Research/dPL/Data/GBEHM',
                  basinFile='E:/Research/dPL/Data/GIS/watershed_sryr.shp',
                  tRange=['2000-1-1', '2019-12-31'],
                  stations=['TNH', 'JUG', 'MAQ', 'MET', 'JIM'])

    out_dict = OrderedDict(Qs=model.Qs, ET=model.ET, snowDepth=model.snowDep, soiwice=model.soiwice, soiwliq=model.soiwliq)
    with open('../checkpoints/GBEHM.pkl', 'wb') as f:
        f.write(pickle.dumps(out_dict))
"""Use the University of Hawaii PyCurrents spectral analysis function to calculate rotary spectra and save the results in data/spectra."""

from cartopy.feature import NaturalEarthFeature
import cartopy.crs as ccrs
import numpy as np
import os
import pandas as pd
import proplot as pplt
from pycurrents.num import spectra

def bathymetric_regions(buoy, buoy_data):
    """Returns bathymetric region based on the following criteria:
    Nansen Basin: deeper than 3000 m, north of 81.5"""
    lat = buoy_data[buoy]['latitude']
    lon = buoy_data[buoy]['longitude']
    depth = buoy_data[buoy]['depth']
    
    nansen = ((lat > 82) & (lon > 0)) & ((lat < 85) & (depth < -3000))
    yermak = ((lat < 84) & (lon > 0)) & ((lat > 80) & (depth > -1000))
    channel = ((lat < 82.5) & (lat > 75)) & ((lon < 2) & (depth < -1500))
    shelf = (((lat < 83) & (lat > 65)) & ((lon < 0) & (lon > -25)) & (depth > -500))
    
    channel = channel & ~yermak
    return {'NB': nansen, 'YP': yermak, 'GC': channel, 'GS': shelf}

tidal_constituents = pd.Series(
    [0.92954, 0.99726, 1.00274, 1.89598, 1.93227, 2.0],
    index=['O1', 'P1', 'K1', 'N2', 'M2', 'S2']
)
f = lambda latitude: 2*2*np.pi/(24*3600)*np.sin(np.deg2rad(latitude))
fperday = lambda latitude: 24 * 3600 * f(latitude) / (2*np.pi)

lpfilter = 35 

def get_spectral_df(buoy_data_subset, uvar='u', vvar='v'):
    """Assumptions about buoy_data_subset:
    1. All time series are the same length
    2. All time series should be included in the dataframes
    3. NaNs have already been taken care off"""
    spectral_results = {}
    for buoy in buoy_data_subset:
        u = buoy_data_subset[buoy][uvar] - buoy_data_subset[buoy][uvar].rolling(
            lpfilter, center=True, win_type='bartlett').mean()
        v = buoy_data_subset[buoy][vvar] - buoy_data_subset[buoy][vvar].rolling(
            lpfilter, center=True, win_type='bartlett').mean()
        U = u + 1j*v

        
        #        lat = buoy_data[buoy].loc[ts, 'latitude'].median()
        s = spectra.spectrum(U.dropna(), nfft=None, dt=1/24, 
                             window='hanning',
                             smooth=3)

        spectral_results[buoy] = s
        spectral_results[buoy]['lat'] = buoy_data_subset[buoy]['latitude']
    df_cw = pd.DataFrame({b: spectral_results[b]['cwpsd'] for b in spectral_results},
                 index=spectral_results[buoy]['cwfreqs'])
    df_cw.index.names = ['frequency']
    df_ccw = pd.DataFrame({b: spectral_results[b]['ccwpsd'] for b in spectral_results},
                 index=spectral_results[buoy]['ccwfreqs'])
    df_ccw.index.names = ['frequency']
    return df_cw, df_ccw

#### Read in data
dataloc = '../data/hourly_merged_buoy_data/'
#buoy_data = {file.split('_')[2].split('.')[0]:
buoy_data = {file.split('.')[0]:
                 pd.read_csv(dataloc + file,
                              index_col=0, parse_dates=True)
             for file in os.listdir(dataloc) if file.split('.')[-1] == 'csv'}

#### Gather tracks from regions
buoy_subsets = {'NB': {},
                'YP': {},
                'GC': {},
                'GS': {}
               }
dt = pd.to_timedelta('20D')
min_obs = 24*18 # At least 20 days of data
for buoy in buoy_data:
    regions = bathymetric_regions(buoy, buoy_data)
    for reg in regions:
        if np.sum(regions[reg]) > min_obs:
            idx = regions[reg]
            begin = idx[idx].index[0]
            end = begin + dt
            buoy_df = buoy_data[buoy].loc[slice(begin, end)]            
            ref_index = pd.Series(np.nan, index=pd.date_range(begin, end, freq='1H'), name='new_index')
            if (len(buoy_df) != len(ref_index) & len(buoy_df)) > (0.95*len(ref_index)):
                buoy_df = buoy_df.merge(new_index, left_index=True, right_index=True).interpolate().fillna(0)

            if len(buoy_df) == len(ref_index):
                buoy_subsets[reg][buoy] = buoy_df
# For greenland channel, remove tracks that 
buoy_subsets['GC'] = {b: buoy_subsets['GC'][b] for b in buoy_subsets['GC'] if \
                        buoy_subsets['GC'][b]['depth'].max() < -1000}            


#### Plot Spectra ####
for reg in buoy_subsets:
    df_cw, df_ccw = get_spectral_df(buoy_subsets[reg])

    df_cw.to_csv('../data/spectra/' + reg + '_CW.csv')
    df_ccw.to_csv('../data/spectra/' + reg + '_CCW.csv')
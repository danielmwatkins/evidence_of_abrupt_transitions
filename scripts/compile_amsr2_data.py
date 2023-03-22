"""Reads in the daily 25 km AMSR2 data obtained from NSIDC and compiles the files
into a single netcdf file."""

import xarray as xr
import pandas as pd
import numpy as np
import os
import h5py

sic_dataloc = '../../data/amsr2/' # Points to a directory with the daily 25km files. Not included in archive.
files = [f for f in os.listdir(sic_dataloc) if f != '.DS_Store']
files = [f for f in files if f.split('.')[-1] == 'he5']
files.sort()

cell_area = xr.open_dataset(sic_dataloc + 'NSIDC0771_CellArea_PS_N25km_v1.0.nc')
latlon_grid = xr.open_dataset(sic_dataloc + 'NSIDC0771_LatLon_PS_N25km_v1.0.nc')

with h5py.File(sic_dataloc + files[0]) as ds1:
    lats = ds1['HDFEOS']['GRIDS']['NpPolarGrid25km']['lat'][:, :]
    lons = ds1['HDFEOS']['GRIDS']['NpPolarGrid25km']['lon'][:, :]

dates = [pd.to_datetime(f.split('.')[0].split('_')[-1], format='%Y%m%d') for f in files]
data = []
for file in files:
    with h5py.File(sic_dataloc + file) as ds:
        data.append(ds['HDFEOS']['GRIDS']['NpPolarGrid25km']['Data Fields']['SI_25km_NH_ICECON_DAY'][:,:])

ds = xr.Dataset({'sea_ice_concentration': (('time', 'y', 'x'), data),
                 'latitude': (('y', 'x'), latlon_grid['latitude'].data),
                 'longitude': (('y', 'x'), latlon_grid['longitude'].data)},
           coords={'time': ('time', dates),
                   'x': ('x', latlon_grid['x'].data),
                   'y': ('y', latlon_grid['y'].data)
                 })

ds.attrs = {'sea_ice_concentration': '0: Open Water\n110: Missing\n120: Land\n1-100: Sea ice concentration',
            'crs': 'NSIDC Polar Stereographic North Pole 25 km'}
ds.to_netcdf('../data/amsr2_sea_ice_concentration.nc',
             engine='netcdf4',
             encoding={var: {'zlib': True} for var in 
                       ['x', 'y', 'sea_ice_concentration',
                        'latitude', 'longitude']})
"""Takes cleaned MOSAiC data and adds AMSR2 sea ice concentration, ERA5 winds, and IBCAO depths interpolated to the buoy position. Pre-requisites are that the buoy tracks are
already quality controlled and interpolated, that the AMSR2 data has been compiled
and that the ERA5 data has been downloaded."""

import os
import numpy as np
import pandas as pd
import pyproj
import sys
import warnings
import xarray as xr
from scipy.interpolate import interp2d
from drifter import compute_velocity

warnings.simplefilter(action='ignore', category=FutureWarning)

#### Specify locations for data.
# External datasets not included in archive are stored in a lower 
drift_tracks_loc = '../data/interpolated_tracks/'
sic_loc = '../external_data/amsr2_sea_ice_concentration.nc'
depth_loc = '../external_data/IBCAO_v4_2_400m.nc' 
era5_loc = '../external_data/era5_winds_hourly.nc'
save_loc = '../data/'

# Only load spring/summer data
begin_time = '2020-04-30 00:00'
end_time = '2020-10-01 00:00'

def sic_along_track(position_data, sic_data):
    """Uses the xarray advanced interpolation to get along-track sic
    via nearest neighbors."""
    # Sea ice concentration uses NSIDC NP Stereographic
    crs0 = pyproj.CRS('WGS84')
    crs1 = pyproj.CRS('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')
    transformer_stere = pyproj.Transformer.from_crs(crs0, crs_to=crs1, always_xy=True)
    
    sic = pd.Series(data=np.nan, index=position_data.index)
    
    for date, group in position_data.groupby(position_data.datetime.dt.date):
        x_stere, y_stere = transformer_stere.transform(
            group.longitude, group.latitude)
        
        x = xr.DataArray(x_stere, dims="z")
        y = xr.DataArray(y_stere, dims="z")
        SIC = sic_data.sel(time=date.strftime('%Y-%m-%d'))['sea_ice_concentration'].interp(
            {'x': x,
             'y': y}, method='nearest').data

        sic.loc[group.index] = np.round(SIC.T, 3)
    sic[sic > 100] = np.nan
    return sic

def depth_along_track(position_data, depth_data):
    """Uses the xarray advanced interpolation to get along-track depth
    via nearest neighbor interpolation."""

    # Transform lon/lat to IBCAO x/y
    crs0 = pyproj.CRS('WGS84')
    crs1 = pyproj.CRS('epsg:3996') # IBCAO Polar Stereographic
    lon = position_data.longitude
    lat = position_data.latitude

    z_grid = np.zeros(lon.shape)

    transformer_ll2ps = pyproj.Transformer.from_crs(crs0, crs_to=crs1, always_xy=True)

    x_grid, y_grid = transformer_ll2ps.transform(lon, lat)

    ds_sel = depth_data.sel(x=slice(np.nanmin(x_grid), np.nanmax(x_grid)),
                          y=slice(np.nanmin(y_grid), np.nanmax(y_grid)))

    x = np.array(ds_sel.x.data).astype(float)
    y = np.array(ds_sel.y.data).astype(float)       

    x_grid = xr.DataArray(x_grid, dims=('time'))
    y_grid = xr.DataArray(y_grid, dims=('time'))

    z_interp = ds_sel['z'].astype(np.float64).interp({'x': x_grid, 'y': y_grid}, method='linear')
    return z_interp.data

def era5_uv_along_track(position_data, uv_data):
    """Uses the xarray advanced interpolation to get along-track era5 winds.
    Uses nearest neighbor for now for speed."""
    
    uv = pd.DataFrame(data=np.nan, index=position_data.index, columns=['u_wind', 'v_wind'])
    
    for date, group in position_data.groupby(position_data.datetime):
        
        x = xr.DataArray(group.longitude, dims="z")
        y = xr.DataArray(group.latitude, dims="z")
        U = uv_data.sel(time=date)['u10'].interp(
            {'longitude': x,
             'latitude': y}, method='nearest').data
        V = uv_data.sel(time=date)['v10'].interp(
            {'longitude': x,
             'latitude': y}, method='nearest').data

        uv.loc[group.index, 'u_wind'] = np.round(U.T, 3)
        uv.loc[group.index, 'v_wind'] = np.round(V.T, 3)

    return uv

# Load buoy data
files = os.listdir(drift_tracks_loc)
files = [f for f in files if f.split('.')[-1] == 'csv']
files = [f for f in files if f.split('_')[0] != 'DN']
buoy_data = {file.replace('.csv', '').split('_')[-1]: 
             pd.read_csv(drift_tracks_loc + file, index_col=0,
                         parse_dates=True).loc[slice(begin_time, end_time)]
             for file in files}
buoy_data = {b: buoy_data[b] for b in buoy_data if len(buoy_data[b]) > 24*30}

### Load sea ice concentration and depth data
ds_sic = xr.open_dataset(sic_loc)
ds_depth = xr.open_dataset(depth_loc)
ds_era = xr.open_dataset(era5_loc)

# Join buoy observations into dataframe for faster calculations
all_positions_hourly = []
for buoy in buoy_data:
    df = buoy_data[buoy].loc[:, ['longitude', 'latitude']].copy()
    df = compute_velocity(df, rotate_uv=True, method='c')
    df.reset_index(inplace=True)
    df['buoy'] = buoy
    all_positions_hourly.append(df.loc[:, ['buoy', 'datetime', 'longitude', 'latitude',
                                           'u', 'v', 'speed']])
all_positions_hourly = pd.concat(all_positions_hourly)
all_positions_hourly = all_positions_hourly.loc[(all_positions_hourly.datetime.dt.month > 4) &
                                  (all_positions_hourly.datetime.dt.year > 2019)]
all_positions_hourly.reset_index(drop=True, inplace=True)

all_positions_daily = []
for buoy in buoy_data:
    df = buoy_data[buoy].loc[:, ['longitude', 'latitude']]
    df = df.loc[df.index.hour == 12].copy()
    df = compute_velocity(df, rotate_uv=True, method='c')
    df.reset_index(inplace=True)
    df['buoy'] = buoy
    all_positions_daily.append(df.loc[:, ['buoy', 'datetime', 'longitude', 'latitude',
                                           'u', 'v', 'speed']])
all_positions_daily = pd.concat(all_positions_daily)
all_positions_daily = all_positions_daily.loc[(all_positions_daily.datetime.dt.month > 4) &
                                  (all_positions_daily.datetime.dt.year > 2019)]
all_positions_daily.reset_index(drop=True, inplace=True)

#### Interpolate to the buoy positions ####
# Depth
all_positions_hourly['depth'] = depth_along_track(all_positions_hourly, ds_depth)
all_positions_daily['depth'] = depth_along_track(all_positions_daily, ds_depth)

# Only do SIC for the daily data
all_positions_daily['sea_ice_concentration'] = sic_along_track(all_positions_daily, ds_sic)

# ERA5 winds
print('Loading ERA5 winds')
lonmin = all_positions_hourly.longitude.min()
latmin = all_positions_hourly.latitude.min()
lonmax = all_positions_hourly.longitude.max()
latmax = all_positions_hourly.latitude.max()
ds_sel = ds_era.sel(latitude=slice(latmax, latmin), longitude=slice(lonmin, lonmax)).load()
print('Done')

# Only do wind speed for the hourly data (for daily, need to take median)
all_positions_hourly.loc[:, ['u_wind', 'v_wind']] = era5_uv_along_track(all_positions_hourly, ds_sel)
    
### Mask outside of sea ice region and save results
buoy_data_daily = {}
for buoy, group in all_positions_daily.groupby('buoy'):
    buoy_data_daily[buoy] = group.set_index('datetime')
    
    sic = buoy_data_daily[buoy].sea_ice_concentration
    if np.any(sic < 1):
        last = sic[sic < 1].index[0]
    else:
        last = sic.index[-1]
    ts = slice('2020-05-01', last)
    
    buoy_data_daily[buoy].loc[ts].to_csv(save_loc + 'daily_merged_buoy_data/' + buoy + '.csv')
    
for buoy, group in all_positions_hourly.groupby('buoy'):
    buoy_df = group.set_index('datetime')
    sic = buoy_data_daily[buoy].sea_ice_concentration
    if np.any(sic < 15):
        last = sic[sic < 15].index[0]
    else:
        last = sic.index[-1]
    ts = slice('2020-05-01', last)
    if buoy == '2020P225':
        # Extend the drift track for 2020P225 into open water, this will be 
        # indicated with dashed line on map. 
        ts = slice('2020-05-01', '2020-09-01') 
    buoy_df.loc[ts].to_csv(save_loc + 'hourly_merged_buoy_data/' + buoy + '.csv')


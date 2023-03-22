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

warnings.simplefilter(action='ignore', category=FutureWarning)

#### Specify locations for data.
# External datasets not included in archive are stored in a lower 
drift_tracks_loc = '../data/interpolated_tracks/'
sic_loc = '../external_data/amsr2_sea_ice_concentration.nc'
depth_loc = '../external_data/IBCAO_v4_2_400m.nc' 
era5_loc = '../external_data/era5_winds.nc'
save_loc = '../data/'

# Only load spring/summer data
begin_time = '2020-04-30 00:00'
end_time = '2020-10-01 00:00'

def compute_velocity(buoy_df, date_index=True, rotate_uv=False, method='c'):
    """Computes buoy velocity and (optional) rotates into north and east directions.
    If x and y are not in the columns, projects lat/lon onto stereographic x/y prior
    to calculating velocity. Rotate_uv moves the velocity into east/west. Velocity
    calculations are done on the provided time index. Results will not necessarily 
    be reliable if the time index is irregular. With centered differences, values
    near endpoints are calculated as forward or backward differences.
    
    Options for method
    forward (f): forward difference, one time step
    backward (b): backward difference, one time step
    centered (c): 3-point centered difference
    forward_backward (fb): minimum of the forward and backward differences
    """
    buoy_df = buoy_df.copy()
    
    if date_index:
        date = pd.Series(pd.to_datetime(buoy_df.index.values), index=pd.to_datetime(buoy_df.index))
    else:
        date = pd.to_datetime(buoy_df.date)
        
    delta_t_next = date.shift(-1) - date
    delta_t_prior = date - date.shift(1)
    min_dt = pd.DataFrame({'dtp': delta_t_prior, 'dtn': delta_t_next}).min(axis=1)

    # bwd endpoint means the next expected obs is missing: last data before gap
    bwd_endpoint = (delta_t_prior < delta_t_next) & (np.abs(delta_t_prior - delta_t_next) > 2*min_dt)
    fwd_endpoint = (delta_t_prior > delta_t_next) & (np.abs(delta_t_prior - delta_t_next) > 2*min_dt)
    
    if 'x' not in buoy_df.columns:
        projIn = 'epsg:4326' # WGS 84 Ellipsoid
        projOut = 'epsg:3413' # NSIDC North Polar Stereographic
        transformer = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)

        lon = buoy_df.longitude.values
        lat = buoy_df.latitude.values

        x, y = transformer.transform(lon, lat)
        buoy_df['x'] = x
        buoy_df['y'] = y
    
    if method in ['f', 'forward']:
        dt = (date.shift(-1) - date).dt.total_seconds().values
        dxdt = (buoy_df['x'].shift(-1) - buoy_df['x'])/dt
        dydt = (buoy_df['y'].shift(-1) - buoy_df['y'])/dt

    elif method in ['b', 'backward']:
        dt = (date - date.shift(1)).dt.total_seconds()
        dxdt = (buoy_df['x'] - buoy_df['x'].shift(1))/dt
        dydt = (buoy_df['y'] - buoy_df['y'].shift(1))/dt

    elif method in ['c', 'fb', 'centered', 'forward_backward']:
        fwd_df = compute_velocity(buoy_df.copy(), date_index=date_index, method='forward')
        bwd_df = compute_velocity(buoy_df.copy(), date_index=date_index, method='backward')

        fwd_dxdt, fwd_dydt = fwd_df['u'], fwd_df['v']
        bwd_dxdt, bwd_dydt = bwd_df['u'], bwd_df['v']
        
        if method in ['c', 'centered']:
            dt = (date.shift(-1) - date.shift(1)).dt.total_seconds()
            dxdt = (buoy_df['x'].shift(-1) - buoy_df['x'].shift(1))/dt
            dydt = (buoy_df['y'].shift(-1) - buoy_df['y'].shift(1))/dt
        else:
            dxdt = np.sign(bwd_dxdt)*np.abs(pd.DataFrame({'f': fwd_dxdt, 'b':bwd_dxdt})).min(axis=1)
            dydt = np.sign(bwd_dxdt)*np.abs(pd.DataFrame({'f': fwd_dydt, 'b':bwd_dydt})).min(axis=1)

        dxdt.loc[fwd_endpoint] = fwd_dxdt.loc[fwd_endpoint]
        dxdt.loc[bwd_endpoint] = bwd_dxdt.loc[bwd_endpoint]
        dydt.loc[fwd_endpoint] = fwd_dydt.loc[fwd_endpoint]
        dydt.loc[bwd_endpoint] = bwd_dydt.loc[bwd_endpoint]
    
    if rotate_uv:
        # Unit vectors
        buoy_df['Nx'] = 1/np.sqrt(buoy_df['x']**2 + buoy_df['y']**2) * -buoy_df['x']
        buoy_df['Ny'] = 1/np.sqrt(buoy_df['x']**2 + buoy_df['y']**2) * -buoy_df['y']
        buoy_df['Ex'] = 1/np.sqrt(buoy_df['x']**2 + buoy_df['y']**2) * -buoy_df['y']
        buoy_df['Ey'] = 1/np.sqrt(buoy_df['x']**2 + buoy_df['y']**2) * buoy_df['x']

        buoy_df['u'] = buoy_df['Ex'] * dxdt + buoy_df['Ey'] * dydt
        buoy_df['v'] = buoy_df['Nx'] * dxdt + buoy_df['Ny'] * dydt

        # Calculate angle, then change to 360
        heading = np.degrees(np.angle(buoy_df.u.values + 1j*buoy_df.v.values))
        heading = (heading + 360) % 360
        
        # Shift to direction from north instead of direction from east
        heading = 90 - heading
        heading = (heading + 360) % 360
        buoy_df['bearing'] = heading
        buoy_df['speed'] = np.sqrt(buoy_df['u']**2 + buoy_df['v']**2)
        buoy_df.drop(['Nx', 'Ny', 'Ex', 'Ey'], axis=1, inplace=True)
        
    else:
        buoy_df['u'] = dxdt
        buoy_df['v'] = dydt            
        buoy_df['speed'] = np.sqrt(buoy_df['v']**2 + buoy_df['u']**2)    

    return buoy_df


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
    df = buoy_data[buoy].loc[:, ['longitude', 'latitude']].resample('1D').median()
    df = compute_velocity(df, rotate_uv=True, method='f')
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
    if np.any(sic < 1):
        last = sic[sic < 1].index[0]
    else:
        last = sic.index[-1]
    ts = slice('2020-05-01', last)
    if buoy == '2020P225':
        # Extend the drift track for 2020P225 into open water, this will be 
        # indicated with dashed line on map. 
        ts = slice('2020-05-01', '2020-09-01') 
    buoy_df.loc[ts].to_csv(save_loc + 'hourly_merged_buoy_data/' + buoy + '.csv')


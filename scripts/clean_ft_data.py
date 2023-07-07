"""Reads raw IFT csv files from data/floe_tracker/parsed/, resamples to daily resolution,
and adds ERA5 winds. Results for individual years are saved in data/floe_tracker/interpolated
and a single dataframe with wind speed, drift speed ratio, and turning angle for all
the years are saved in data/floe_tracker.

TBD: reduce file size by decreasing the number of saved significant figures
"""

import numpy as np
import pandas as pd
import pyproj 
from metpy.units import units
import metpy.calc as mcalc
import xarray as xr
from drifter import compute_velocity
from scipy.interpolate import interp1d

saveloc = '../data/floe_tracker/interpolated/'
saveloc_single = '../data/floe_tracker/'
dataloc = '../data/floe_tracker/parsed/'

# Location of folder with ERA5 data. I saved the ERA5 data with 
# a file structure of era5_dataloc/YYYY/era5_uvmsl_daily_mean_YYYY-MM-01.nc
era5_dataloc = '../external_data/era5_daily/'

def regrid_floe_tracker(group, datetime_grid):
    """Estimate the displacement in two stages.
    First, get a best estimate for the position at midnight
    for each day using interpolation. If at least two
    datapoints are available, then calculate the displacement.
    
    Intended use case is to group by floe_id and apply get_displacements
    """
    group.set_index('date', inplace=True)
    group = group.sort_index()
    begin = group.index.min()
    end = group.index.max()

    if len(datetime_grid.loc[slice(begin, end)]) > 1:
        t0 = group.index.round('12H').min()
        t1 = group.index.round('12H').max()
        max_extrap = pd.to_timedelta('2H')
        if np.abs(t0 - begin) < max_extrap:
            begin = t0
        if np.abs(t1 - end) < max_extrap:
            end = t1

        X = group[['x', 'y', 'longitude', 'latitude']].rolling(
            '12H', center=True).mean().T.values
        t_new = datetime_grid.loc[slice(begin, end)].values
        t_seconds = group['t'].values
        Xnew = interp1d(t_seconds, X,
                        bounds_error=False,
                        kind='linear', fill_value='extrapolate')(t_new)
        idx = ~np.isnan(Xnew.sum(axis=0))

        df_new = pd.DataFrame(data=np.round(Xnew.T, 5), 
                              columns=['x', 'y', 'longitude', 'latitude'],
                              index=datetime_grid.loc[slice(begin, end)].index)

        return df_new
    
    else:
        df = pd.DataFrame(data = np.nan, columns=group.columns, index=[begin])
        df.drop(['floe_id', 't'], axis=1, inplace=True)
        return df
    
def era5_uv_along_track(position_data, uv_data):
    """Uses the xarray advanced interpolation to get along-track era5 winds.
    Uses nearest neighbor for now for speed."""
    
    uv = pd.DataFrame(data=np.nan, index=position_data.index, columns=['u_wind', 'v_wind'])
    
    for date, group in position_data.groupby(position_data.date):
        date_midday = date #+ pd.to_timedelta('12H') # This is an issue - should be 
        x = xr.DataArray(group.longitude, dims="z")
        y = xr.DataArray(group.latitude, dims="z")
        U = uv_data.sel(time=date_midday)['u10'].interp(
            {'longitude': x,
             'latitude': y}, method='nearest').data
        V = uv_data.sel(time=date_midday)['v10'].interp(
            {'longitude': x,
             'latitude': y}, method='nearest').data

        uv.loc[group.index, 'u_wind'] = np.round(U.T, 3)
        uv.loc[group.index, 'v_wind'] = np.round(V.T, 3)

    return uv

ft_df_raw = {}
for year in range(2003, 2021):
    df = pd.read_csv(
        dataloc + 'floe_tracker_raw_' + str(year) + '.csv',
        index_col=None).dropna()
    
#     if year == 2020:
#         # The images from 2020 are stretched in the y direction. This is a simple fix 
#         # that gets them pretty close to correct.
#         left=200703.99999999994
#         bottom=-2009088.0
#         right=1093632.0
#         top=-317440.0
#         adjustment = 63.8e3
#         A = ((top - bottom) + adjustment)/(top - bottom)
#         B = top * (1 - A)
#         df['y'] = A*df['y'] + B
#         source_crs = 'epsg:3413'
#         to_crs = 'WGS84'
#         ps2ll = pyproj.Transformer.from_crs(source_crs, to_crs, always_xy=True)
#         lon, lat = ps2ll.transform(df['x'], df['y'])

#         df['longitude'] = np.round(lon, 5)
#         df['latitude'] = np.round(lat, 5)
    ft_df_raw[year] = df
    
ft_df_raw = pd.concat(ft_df_raw)
ft_df_raw.index.names = ['year', 'd1']
ft_df_raw = ft_df_raw.reset_index().drop(['d1'], axis=1)
ft_df_raw['floe_id'] = [str(y) + '_' + str(fi).zfill(4) for y, fi in zip(ft_df_raw['year'], ft_df_raw['floe_id'])]
ft_df_raw['date'] = pd.to_datetime(ft_df_raw['datetime'].values)
print('Number of observations:', len(ft_df_raw))

floe_tracker_results = {}

for year, year_group in ft_df_raw.groupby(ft_df_raw.date.dt.year):
    ref_time = pd.to_datetime(str(year) + '-01-01 00:00')
    date_grid = pd.date_range(str(year) + '-04-01 00:00', str(year) + '-09-30 00:00', freq='1D')
    date_grid += pd.to_timedelta('12H')
    t_grid = (date_grid - ref_time).total_seconds()
    year_group['t'] = (year_group['date'] - ref_time).dt.total_seconds()
    datetime_grid = pd.Series(t_grid, index=date_grid)
    
    results = {}
    for floe_id, group in year_group.groupby('floe_id'):
        if group['date'].dt.month.min() > 2:
            df_regrid = regrid_floe_tracker(group, datetime_grid=datetime_grid)
            if np.any(df_regrid.notnull()):
                results[floe_id] = df_regrid

    floe_tracker_results[year] = pd.concat(results)
    floe_tracker_results[year].index.names = ['floe_id', 'date']
    floe_tracker_results[year].reset_index(inplace=True)
    floe_tracker_results[year] = floe_tracker_results[year].loc[:, ['date', 'floe_id', 'x', 'y', 'longitude', 'latitude']]
    
    floe_tracker_results[year] = floe_tracker_results[year].groupby('floe_id', group_keys=False).apply(
        compute_velocity, date_index=False, rotate_uv=True, method='f').dropna()
    
    # Add ERA5 wind
    floe_tracker_results[year][['u_wind', 'v_wind']] = np.nan
        
    for month, data in floe_tracker_results[year].groupby(floe_tracker_results[year].date.dt.month):
       # Depending on the file name used for the ERA5 data this section will need to be adjusted.
        with xr.open_dataset(era5_dataloc + str(year) + '/' + \
                             'era5_uvmsl_daily_mean_' + \
                             str(year) + '-' + str(month).zfill(2) + '-01.nc') as ds_era:
            floe_tracker_results[year].loc[
                data.index, ['u_wind', 'v_wind']] = era5_uv_along_track(data, ds_era)    

    floe_tracker_results[year].to_csv(saveloc + '/floe_tracker_interp_' + str(year) + '.csv')
    
ft_df = pd.concat(floe_tracker_results)
ft_df = ft_df.loc[(ft_df.speed > 0.02) & (ft_df.speed < 1.5)]
ft_df['wind_speed'] = (ft_df['u_wind']**2 + ft_df['v_wind']**2)**0.5

wind_bearing = mcalc.wind_direction(ft_df['u_wind'].values * units('m/s'),
                 ft_df['v_wind'].values * units('m/s'), convention='to')
ice_bearing = mcalc.wind_direction(ft_df['u'].values * units('m/s'),
                 ft_df['v'].values * units('m/s'), convention='to')
delta = np.deg2rad(ice_bearing.magnitude) - np.deg2rad(wind_bearing.magnitude)
ft_df['turning_angle'] = pd.Series(np.rad2deg(np.arctan2(np.sin(delta), np.cos(delta))), index=ft_df.index)

ft_df['drift_speed_ratio'] = ft_df['speed'] / ft_df['wind_speed']
length = ft_df.groupby('floe_id').apply(lambda x: len(x))
print('Number of distinct floes:', len(length))
print('Number of velocity estimates:', len(ft_df))
print('Median trajectory length:', length.median())
ft_df.to_csv(saveloc_single + 'ft_with_wind.csv')

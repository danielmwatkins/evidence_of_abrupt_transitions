"""Reads raw IFT csv files from data/floe_tracker/parsed/, resamples to daily resolution,
and adds ERA5 winds. Results for individual years are saved in data/floe_tracker/interpolated
and a single dataframe with wind speed, drift speed ratio, and turning angle for all
the years are saved in data/floe_tracker.

TBD: reduce file size by decreasing the number of saved significant figures
"""

import numpy as np
import pandas as pd
import pyproj 
import proplot as pplt
from metpy.units import units
import metpy.calc as mcalc
from scipy.interpolate import interp1d
import xarray as xr

saveloc = '../data/floe_tracker/interpolated/'
saveloc_single = '../data/floe_tracker/'
dataloc = '../data/floe_tracker/parsed/'
# Location of folder with ERA5 data. I saved the ERA5 data with 
# a file structure of era5_dataloc/YYYY/era5_uvmsl_daily_mean_YYYY-MM-01.nc
era5_dataloc = '../../../mosaic_drift_climatology/data/era5/'

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
        date = pd.to_datetime(buoy_df.datetime)
        
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

def regrid_floe_tracker(group, datetime_grid):
    """Estimate the displacement in two stages.
    First, get a best estimate for the position at midnight
    for each day using interpolation. If at least two
    datapoints are available, then calculate the displacement.
    
    Intended use case is to group by floe_id and apply get_displacements
    """
    group.set_index('datetime', inplace=True)
    begin = group.index.min()
    end = group.index.max()

    if len(datetime_grid.loc[slice(begin, end)]) > 1:
        t0 = group.index.round('24H').min()
        t1 = group.index.round('24H').max()
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
    
    for date, group in position_data.groupby(position_data.datetime):
        date_midday = date + pd.to_timedelta('12H')
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
    ft_df_raw[year] = pd.read_csv(
        dataloc + 'floe_tracker_raw_' + str(year) + '.csv',
        index_col=None).dropna()
    
ft_df_raw = pd.concat(ft_df_raw)
ft_df_raw.index.names = ['year', 'd1']
ft_df_raw = ft_df_raw.reset_index().drop(['d1'], axis=1)
ft_df_raw['floe_id'] = [str(y) + '_' + str(fi) for y, fi in zip(ft_df_raw['year'], ft_df_raw['floe_id'])]
ft_df_raw['datetime'] = pd.to_datetime(ft_df_raw['datetime'].values)


floe_tracker_results = {}

for year, year_group in ft_df_raw.groupby(ft_df_raw.datetime.dt.year):
    ref_time = pd.to_datetime(str(year) + '-01-01 00:00')
    date_grid = pd.date_range(str(year) + '-04-01 00:00', str(year) + '-09-30 00:00', freq='1D')
    t_grid = (date_grid - ref_time).total_seconds()
    year_group['t'] = (year_group['datetime'] - ref_time).dt.total_seconds()
    datetime_grid = pd.Series(t_grid, index=date_grid)
    
    results = {}
    for floe_id, group in year_group.groupby('floe_id'):
        if group['datetime'].dt.month.min() > 2:
            df_regrid = regrid_floe_tracker(group, datetime_grid=datetime_grid)
            if np.any(df_regrid.notnull()):
                results[floe_id] = df_regrid

    floe_tracker_results[year] = pd.concat(results)
    floe_tracker_results[year].index.names = ['floe_id', 'datetime']
    floe_tracker_results[year].reset_index(inplace=True)
    floe_tracker_results[year] = floe_tracker_results[year].loc[:, ['datetime', 'floe_id', 'x', 'y', 'longitude', 'latitude']]
    
    floe_tracker_results[year] = floe_tracker_results[year].groupby('floe_id', group_keys=False).apply(
        compute_velocity, date_index=False, rotate_uv=True, method='f').dropna()
    
    # Add ERA5 wind
    floe_tracker_results[year][['u_wind', 'v_wind']] = np.nan
    for month, data in floe_tracker_results[year].groupby(floe_tracker_results[year].datetime.dt.month):
       # Depending on the file name used for the ERA5 data this section will need to be adjusted.
        with xr.open_dataset(era5_dataloc + str(year) + \
                             '/era5_uvmsl_daily_mean_' + \
                             str(year) + '-' + str(month).zfill(2) + '-01.nc') as ds_era:
            floe_tracker_results[year].loc[data.index, ['u_wind', 'v_wind']] = era5_uv_along_track(data, ds_era)    
    floe_tracker_results[year].to_csv(saveloc + '/floe_tracker_interp_' + str(year) + '.csv')
    
ft_df = pd.concat(floe_tracker_results)
ft_df = ft_df.loc[ft_df.speed > 0.02]
ft_df['wind_speed'] = (ft_df['u_wind']**2 + ft_df['v_wind']**2)**0.5

wind_bearing = mcalc.wind_direction(ft_df['u_wind'].values * units('m/s'),
                 ft_df['v_wind'].values * units('m/s'), convention='to')
ice_bearing = mcalc.wind_direction(ft_df['u'].values * units('m/s'),
                 ft_df['v'].values * units('m/s'), convention='to')
delta = np.deg2rad(ice_bearing.magnitude) - np.deg2rad(wind_bearing.magnitude)
ft_df['turning_angle'] = pd.Series(np.rad2deg(np.arctan2(np.sin(delta), np.cos(delta))), index=ft_df.index)

ft_df['drift_speed_ratio'] = ft_df['speed'] / ft_df['wind_speed']

ft_df.to_csv(saveloc_single + 'ft_with_wind.csv')

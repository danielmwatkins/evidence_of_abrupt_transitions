import cartopy.crs as ccrs
import metpy.calc as mcalc
from metpy.units import units
import numpy as np
import os
import proplot as pplt
import pandas as pd
import scipy.stats as stats
import xarray as xr
import warnings
warnings.simplefilter('ignore')


### Load data
ds_depth = xr.open_dataset('../data/interpolated_depth.nc')

ft_df = pd.read_csv('../data/floe_tracker/ft_with_wind.csv')

#### Load the buoy data
# For this section, we pull in the daily data to match the timing of the IFT observations
# The wind data used for the IFT section is from ERA5 daily averages, hence we 
# read in the hourly interpolated winds and calculated daily means, then we 
# merge those daily average winds to the buoy positions.
dataloc = '../data/daily_merged_buoy_data/'
files = os.listdir(dataloc)
buoy_data = {f.split('.')[0]: pd.read_csv(dataloc + f,
                 parse_dates=True, index_col=0) for f in files if f.split('.')[1] == 'csv'}

dataloc = '../data/hourly_merged_buoy_data/'
files = os.listdir(dataloc)
buoy_data_hourly = {f.split('.')[0]: pd.read_csv(dataloc + f,
                 parse_dates=True, index_col=0).resample('1D').mean() for f in files if f.split('.')[1] == 'csv'}

buoy_df = pd.concat(buoy_data).reset_index().drop('level_0', axis=1)
buoy_df = buoy_df.loc[buoy_df['datetime'] > pd.to_datetime('2020-07-13 00:00')].dropna()

buoy_df_hourly = pd.concat(buoy_data_hourly).reset_index().drop('buoy', axis=1)
buoy_df_hourly.rename({'level_0': 'buoy'}, axis=1, inplace=True)
buoy_df_hourly = buoy_df_hourly.loc[buoy_df_hourly['datetime'] > pd.to_datetime('2020-07-13 00:00')].dropna()

buoy_df = buoy_df.merge(buoy_df_hourly.loc[:, ['buoy', 'datetime', 'u_wind', 'v_wind']],
             on=['buoy', 'datetime'], how='left')
buoy_df['wind_speed'] = (buoy_df['u_wind']**2 + buoy_df['v_wind']**2)**0.5
wind_bearing = mcalc.wind_direction(buoy_df['u_wind'].values * units('m/s'),
                 buoy_df['v_wind'].values * units('m/s'), convention='to')
ice_bearing = mcalc.wind_direction(buoy_df['u'].values * units('m/s'),
                 buoy_df['v'].values * units('m/s'), convention='to')
delta = np.deg2rad(ice_bearing.magnitude) - np.deg2rad(wind_bearing.magnitude)
buoy_df['turning_angle'] = pd.Series(np.rad2deg(np.arctan2(np.sin(delta), np.cos(delta))), index=buoy_df.index)

buoy_df['drift_speed_ratio'] = buoy_df['speed'] / buoy_df['wind_speed']




### Prepare histograms and binned statistics 
x = ft_df.longitude
y = ft_df.latitude
longrid = np.arange(-30, 15, 0.5)
latgrid = np.arange(65, 85, 0.25)
nmin = 20
lon_c = 0.5*(longrid[1:] + longrid[:-1])
lat_c = 0.5*(latgrid[1:] + latgrid[:-1])

# Estimate wind component
model_alpha = 0.02
model_theta = 20
U_est = model_alpha * np.exp(1j*np.deg2rad(model_theta))*(ft_df['u_wind'] + 1j*ft_df['v_wind'])
u_est = pd.Series(np.real(U_est), index=ft_df.index)
v_est = pd.Series(np.imag(U_est), index=ft_df.index)
ft_df['u_est'] = u_est
ft_df['v_est'] = v_est
ft_df['u_res'] = ft_df['u'] - ft_df['u_est']
ft_df['v_res'] = ft_df['v'] - ft_df['v_est']

# Variables:
# N, u_median, v_median, wind_speed, u_res, v_res
hist2d = np.histogram2d(ft_df['longitude'],
               ft_df['latitude'],
              bins=[longrid, latgrid])
df_hist = pd.DataFrame(hist2d[0], index=lon_c, columns=lat_c)

sel = ft_df.wind_speed > 0
u_median, xedges, yedges, binnumber = stats.binned_statistic_2d(
    x[sel], y[sel], values=ft_df.u[sel], statistic='median', 
    bins=[longrid, latgrid])
v_median, xedges, yedges, binnumber = stats.binned_statistic_2d(
    x[sel], y[sel], values=ft_df.v[sel], statistic='median', 
    bins=[longrid, latgrid])
wind_speed, xedges, yedges, binnumber = stats.binned_statistic_2d(
    x[sel], y[sel], values=ft_df.wind_speed[sel], statistic='median', 
    bins=[longrid, latgrid])
u_res, xedges, yedges, binnumber = stats.binned_statistic_2d(
    x[sel], y[sel], values=ft_df.u_res[sel], statistic='median', 
    bins=[longrid, latgrid])
v_res, xedges, yedges, binnumber = stats.binned_statistic_2d(
    x[sel], y[sel], values=ft_df.v_res[sel], statistic='median', 
    bins=[longrid, latgrid])
speed_mad, xedges, yedges, binnumber = stats.binned_statistic_2d(
    x[sel], y[sel], values=ft_df.speed[sel], statistic=stats.median_abs_deviation, 
    bins=[longrid, latgrid])
speed, xedges, yedges, binnumber = stats.binned_statistic_2d(
    x[sel], y[sel], values=ft_df.speed[sel], statistic='median', 
    bins=[longrid, latgrid])

u_median = pd.DataFrame(u_median, index=lon_c, columns=lat_c)
v_median = pd.DataFrame(v_median, index=lon_c, columns=lat_c)
u_res = pd.DataFrame(u_res, index=lon_c, columns=lat_c)
v_res = pd.DataFrame(v_res, index=lon_c, columns=lat_c)
speed_mad = pd.DataFrame(speed_mad, index=lon_c, columns=lat_c)
speed = pd.DataFrame(speed, index=lon_c, columns=lat_c)
wind_speed = pd.DataFrame(wind_speed, index=lon_c, columns=lat_c)


# Group variables for ease of plotting
variables = {
    'Observation Counts': {'data': df_hist.where(df_hist > 0),
                           'vmin': 0,
                           'vmax': 200,
                           'clabel': 'N'},
    'Drift Speed': {'data': speed.where(df_hist > nmin),
                    'vmin': 0,
                    'vmax': 0.2,
                    'clabel': 'm/s'},
    'Median Abs. Deviation': {'data': speed_mad.where(df_hist > nmin),
                 'vmin': 0,
                 'vmax': 0.2,
                 'clabel': 'm/s'},
    'Drift Velocity': {'data_u': u_median.where(df_hist > nmin),
                        'data_v': v_median.where(df_hist > nmin),
                        },
    'Wind Speed': {'data': wind_speed.where(df_hist > nmin),
                         'vmin': 0,
                         'vmax': 10,
                         'clabel': 'm/s'},
    'U Residual': {'data': u_res.where(df_hist > nmin),
                 'vmin': -0.15,
                 'vmax': 0.15,
                 'clabel': 'm/s'},
    'V Residual': {'data': v_res.where(df_hist > nmin),
                 'vmin': -0.15,
                 'vmax': 0.15,
                 'clabel': 'm/s'},
    'Residual Velocity': {'data_u': u_res.where(df_hist > nmin),
                           'data_v': v_res.where(df_hist > nmin),
                          }}



#### Plot Ice Floe Tracker maps

pplt.rc.reso = 'med'

# proj = {idx: proj='lcc', width=10, proj_kw={'lon_0': 0}}
fig, axs = pplt.subplots(proj='lcc', width=10, proj_kw={'lon_0': 0}, ncols=4, nrows=3, share=False)
axs.format(land=True, latlim=(70,81),
           lonlim=(-30,10), facecolor='gray1', landzorder=10, lonlabels=True)
for ax in axs:
    ax.contour(ds_depth.longitude,
                ds_depth.latitude,
                ds_depth.z, levels=[-1500,  -500, 0],
           colors=['k', 'k', 'gray1'], ls='-', labels=True, zorder=5, lw=1)
    
for ax, variable in zip(axs, variables):
    if 'Velocity' not in variable:
        if variable in ['U Residual', 'V Residual']:
            ex = 'both'
            cmap = 'spectral_r'
        else:
            ex = 'max'
            cmap = 'spectral_r'
        df = variables[variable]['data']
        ax.format(latlabels=True)
        c = ax.pcolormesh(lon_c, lat_c, df.T, zorder=4,
                          vmin=variables[variable]['vmin'],
                          vmax=variables[variable]['vmax'], N=20,
                          cmap=cmap, extend=ex)
        ax.colorbar(c, label=variables[variable]['clabel'], loc='b')
        ax.format(title=variable, titlesize=10)
        
    else:
        u = variables[variable]['data_u']
        v = variables[variable]['data_v']
        ax.quiver(lon_c[::3], lat_c[::3],
          u.T.loc[::3, ::3],
          v.T.loc[::3, ::3],
          zorder=10, scale=1, width=2/500, headwidth=7, headlength=5, color='r')
        ax.format(title=variable, titlesize=10)

# ax.format(latlabels=True)
for ax in [axs[0,-1], axs[1,-1]]:
    ax.quiver(1, 70.75, 0.2, 0,
          zorder=10, scale=1, width=2/500, headwidth=7, headlength=5, color='r')
    ax.text(1, 71, '20 cm/s', color='r', fontsize=8, transform=ccrs.PlateCarree())
# fig.save('../figures/figure3_abcd.png', dpi=300)


### Plot histograms

# fig, ax = pplt.subplots(ncols=4, share=False)
ax = fig.add_subplots(341)
x = ax[0].hist2d(ft_df['wind_speed'], ft_df['turning_angle'], bins=[np.linspace(0, 15, 50),
                                      np.linspace(-180, 180, 50)],
         cmap='spectral_r')


x = ax[1].hist2d(ft_df['wind_speed'], ft_df['drift_speed_ratio'], bins=[np.linspace(0, 15, 50),
                                      np.linspace(0, 0.2, 100)],
         cmap='spectral_r')
ax[0].axhline(0, color='k', lw=0.5)
# ax[0].format(ylabel='Turning Angle ($\\Theta$)', xlabel='$U_{wind}$', title='Ice Floe Tracker')
# ax[1].format(ylabel='Drift Speed Ratio ($\\alpha$)', xlabel='$U_{wind}$', title='Ice Floe Tracker')


x = ax[2].hist2d(buoy_df['wind_speed'], buoy_df['turning_angle'], bins=[np.linspace(0, 15, 50),
                                      np.linspace(-180, 180, 50)], density=True, cmap='spectral_r')


x = ax[3].hist2d(buoy_df['wind_speed'], buoy_df['drift_speed_ratio'], bins=[np.linspace(0, 15, 50),
                                      np.linspace(0, 0.2, 50)], density=True, cmap='spectral_r')
ax[2].axhline(0, color='k', lw=0.5)
# ax[2].format(ylabel='Turning Angle ($\\Theta$)', xlabel='$U_{wind}$', title='MOSAiC')
# ax[3].format(ylabel='Drift Speed Ratio ($\\alpha$)', xlabel='$U_{wind}$', title='MOSAiC')
fig.format(abc=True)
fig.save('../figures/figure3.png', dpi=300)

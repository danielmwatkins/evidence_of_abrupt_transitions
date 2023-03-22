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

lon_c = 0.5*(longrid[1:] + longrid[:-1])
lat_c = 0.5*(latgrid[1:] + latgrid[:-1])

u_median, xedges, yedges, binnumber = stats.binned_statistic_2d(
    x, y, values=ft_df.u, statistic='median', 
    bins=[longrid, latgrid])
v_median, xedges, yedges, binnumber = stats.binned_statistic_2d(
    x, y, values=ft_df.v, statistic='median', 
    bins=[longrid, latgrid])
v_iqr, xedges, yedges, binnumber = stats.binned_statistic_2d(
    x, y, values=ft_df.v, statistic=stats.iqr, 
    bins=[longrid, latgrid])
u_iqr, xedges, yedges, binnumber = stats.binned_statistic_2d(
    x, y, values=ft_df.u, statistic=stats.iqr, 
    bins=[longrid, latgrid])
utot, xedges, yedges, binnumber = stats.binned_statistic_2d(
    x, y, values=(ft_df.v**2 + ft_df.u**2)**0.5, statistic='median', 
    bins=[longrid, latgrid])
utot_iqr, xedges, yedges, binnumber = stats.binned_statistic_2d(
    x, y, values=(ft_df.v**2 + ft_df.u**2)**0.5, statistic=stats.iqr, 
    bins=[longrid, latgrid])



u_median = pd.DataFrame(u_median, index=lon_c, columns=lat_c)
v_median = pd.DataFrame(v_median, index=lon_c, columns=lat_c)
u_iqr = pd.DataFrame(u_iqr, index=lon_c, columns=lat_c)
v_iqr = pd.DataFrame(v_iqr, index=lon_c, columns=lat_c)
utot = pd.DataFrame(utot, index=lon_c, columns=lat_c)
utot_iqr = pd.DataFrame(utot_iqr, index=lon_c, columns=lat_c)

hist2d = np.histogram2d(ft_df['longitude'],
               ft_df['latitude'],
              bins=[longrid, latgrid])
df_hist = pd.DataFrame(hist2d[0], index=lon_c, columns=lat_c)  


#### Plot Ice Floe Tracker maps
nmin = 20
pplt.rc.reso = 'med'
fig, axs = pplt.subplots(proj='lcc', width=10, proj_kw={'lon_0': 0}, ncols=4, nrows=1, sharey=True)
axs.format(land=True, latlim=(70,81), lonlim=(-30,10), facecolor='gray1', landzorder=10, lonlabels=True)
for ax in axs:
    ax.contour(ds_depth.longitude,
                ds_depth.latitude,
                ds_depth.z, levels=np.arange(-3500, -1, 1000),
           color='k', ls='-', labels=True, zorder=9, lw=1)
    
ax = axs[0,0]
ax.format(latlabels=True)
c = ax.pcolormesh(lon_c, lat_c, df_hist.where(df_hist>0).T, zorder=4, vmin=0, vmax=200, N=10,
                  cmap='spectral_r', extend='max')
ax.colorbar(c, label='N', loc='b')
ax.format(title='Observation Counts', titlesize=10)


ax = axs[0,1]
c = ax.pcolormesh(lon_c, lat_c, utot.where(df_hist>nmin).T, zorder=4, vmin=0, vmax=0.2, N=10,
                  cmap='spectral_r', extend='max')
ax.colorbar(c, label='U (m/s)', loc='b')
ax.format(title='Drift Speed', titlesize=10)


ax = axs[0,2]
ax.quiver(lon_c[::2], lat_c[::2],
          u_median.where(df_hist > nmin).T.loc[::2, ::2],
          v_median.where(df_hist > nmin).T.loc[::2, ::2],
          zorder=10, scale=3, headwidth=10, headlength=5, color='r')

ax.quiver(5, 71, 0.2, 0,
          zorder=10, scale=3, color='r')
ax.text(5, 71.5, '0.2 m/s', color='r', fontsize=5, transform=ccrs.PlateCarree())
ax.format(title='Drift Direction', titlesize=10)

ax = axs[0,3]
c = ax.pcolormesh(lon_c, lat_c, utot_iqr.where(df_hist>nmin).T, zorder=4, vmin=0, vmax=0.15, N=10,
                  cmap='spectral_r', extend='max')
ax.colorbar(c, title='$U_{IQR} + V_{IQR}$ (m/s)', loc='b')
ax.format(title='Drift Speed Total IQR', titlesize=10)
fig.save('../figures/figure3_abcd.png', dpi=300)


### Plot histograms
from metpy.units import units
import metpy.calc as mcalc

fig, ax = pplt.subplots(ncols=4, share=False)
x = ax[0].hist2d(ft_df['wind_speed'], ft_df['turning_angle'], bins=[np.linspace(0, 15, 50),
                                      np.linspace(-180, 180, 50)],
         cmap='spectral_r')


x = ax[1].hist2d(ft_df['wind_speed'], ft_df['drift_speed_ratio'], bins=[np.linspace(0, 15, 50),
                                      np.linspace(0, 0.2, 100)],
         cmap='spectral_r')
ax[0].axhline(0, color='k', lw=0.5)
ax[0].format(ylabel='Turning Angle ($\\Theta$)', xlabel='$U_{wind}$', title='Ice Floe Tracker')
ax[1].format(ylabel='Drift Speed Ratio ($\\alpha$)', xlabel='$U_{wind}$', title='Ice Floe Tracker')


x = ax[2].hist2d(buoy_df['wind_speed'], buoy_df['turning_angle'], bins=[np.linspace(0, 15, 50),
                                      np.linspace(-180, 180, 50)], density=True, cmap='spectral_r')


x = ax[3].hist2d(buoy_df['wind_speed'], buoy_df['drift_speed_ratio'], bins=[np.linspace(0, 15, 50),
                                      np.linspace(0, 0.2, 50)], density=True, cmap='spectral_r')
ax[2].axhline(0, color='k', lw=0.5)
ax[2].format(ylabel='Turning Angle ($\\Theta$)', xlabel='$U_{wind}$', title='MOSAiC')
ax[3].format(ylabel='Drift Speed Ratio ($\\alpha$)', xlabel='$U_{wind}$', title='MOSAiC')
fig.save('../figures/figure3_efgh.png', dpi=300)

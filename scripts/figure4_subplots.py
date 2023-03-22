import cartopy.crs as ccrs
import numpy as np
import os
import pandas as pd
import proplot as pplt
import xarray as xr
import warnings

# There's a RuntimeWarning from shapely raised in the map plotting function. I don't see any 
# negative effects from it so I'm surpressing it here.
warnings.simplefilter('ignore')

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

#### Read in data
dataloc = '../data/hourly_merged_buoy_data/'
buoy_data = {file.split('.')[0]:
                 pd.read_csv(dataloc + file,
                              index_col=0, parse_dates=True)
             for file in os.listdir(dataloc) if file.split('.')[-1] == 'csv'}

ds_depth = xr.open_dataset('../data/interpolated_depth.nc')

df_r2 = pd.read_csv('../data/harmonic_fit/correlation_coefficients.csv', index_col=0)
df_U = pd.read_csv('../data/harmonic_fit/maximum_daily_current.csv', index_col=0)

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
            ref_index = pd.Series(np.nan,
                                  index=pd.date_range(begin, end, freq='1H'),
                                  name='new_index')
            if (len(buoy_df) != len(ref_index) & \
                len(buoy_df)) > (0.95*len(ref_index)):
                buoy_df = buoy_df.merge(new_index,
                                        left_index=True,
                                        right_index=True).interpolate()

            if len(buoy_df) == len(ref_index):
                buoy_subsets[reg][buoy] = buoy_df
                
# For greenland channel, remove tracks that never make it all the way into the
# deeper part of the channel 
buoy_subsets['GC'] = {b: buoy_subsets['GC'][b] for b in buoy_subsets['GC'] if \
                        buoy_subsets['GC'][b]['depth'].max() < -1000}            

#### Plot Regions ####
colors = ['gold', 'tab:cyan', 'plum', 'lime8']

pplt.rc.reso = 'med'
pplt.rc['title.pad'] = 0.1
pplt.rc['title.border'] = False
fig, ax = pplt.subplots(proj='lcc', height=2.5, proj_kw={'lon_0': 0})
ax.format(land=True, latlim=(75,85), lonlim=(-25,20), facecolor='gray1',
          latlabels=True, lonlabels=True, grid=False)
cbar = ax.contourf(ds_depth.longitude,
                ds_depth.latitude,
                ds_depth.z, levels=[-4000, -3500, -3000, -2500,
                              -2000, -1500, -1000, -500,
                              -200, -100, -50, 0],
                cmap='blues8_r',
                extend='both')
h, l = [], []
for region, color in zip(buoy_subsets, colors):
    for buoy in buoy_subsets[region]:
        ax.plot(buoy_subsets[region][buoy].longitude, buoy_subsets[region][buoy].latitude, lw=0.5, color=color)

    h.append(ax.plot([],[], m='s', lw=0, color=color))
    l.append(region)

ax.legend(h, l, loc='ul', ncols=1, alpha=1)
fig.save('../figures/figure4_e.png', dpi=300)

#### Plot Spectra ####
tidal_constituents = pd.Series(
    [0.92954, 0.99726, 1.00274, 1.89598, 1.93227, 2.0],
    index=['O1', 'P1', 'K1', 'N2', 'M2', 'S2']
)
f = lambda latitude: 2*2*np.pi/(24*3600)*np.sin(np.deg2rad(latitude))
fperday = lambda latitude: 24 * 3600 * f(latitude) / (2*np.pi)


fig, axs = pplt.subplots(width=10, height=2.5, ncols=4, nrows=1, spanx=True, sharex=False)
colors = {tide: c['color'] for c, tide in zip(pplt.Cycle('colorblind'), tidal_constituents.index)}



for ax, reg, title in zip(axs, 
        ['NB', 'YP', 'GC', 'GS'],
        ['Nansen Basin', 'Yermak Plateau', 'Channel', 'Shelf']):
    df_cw = pd.read_csv('../data/spectra/' + reg + '_CW.csv', index_col=0)
    df_ccw = pd.read_csv('../data/spectra/' + reg + '_CCW.csv', index_col=0)

    for df, color in zip([df_cw, df_ccw], ['blue', 'red']):
        ax.plot(df.index, df.median(axis=1), color=color, lw=1.5, marker='.', ms=1,
                shadedata=[df.quantile(0.75, axis=1),
                          df.quantile(0.25, axis=1)],
                fadedata=[df.quantile(0.9, axis=1),
                          df.quantile(0.1, axis=1)])

        ax.plot(df.index, df.max(axis=1), color=color, lw=0.5, ls=':')
        ax.plot(df.index, df.min(axis=1), color=color, lw=0.5, ls=':')
        
    df_lat = pd.DataFrame({b: buoy_subsets[reg][b]['latitude']
                           for b in buoy_subsets[reg]})
    f_reg = fperday(df_lat.mean().mean())
    n = df.shape[1]
    ax.format(xlabel='Cycles per day',
              ylabel='$S_U$ (m$^2$s$^{-2}$ per cycle per day)',
              urtitle='n=' + str(n) + '  \nf=' + str(np.round(f_reg,2)))

    for tide in tidal_constituents.index:
        ax.axvline(tidal_constituents[tide], lw=0.5, zorder=0, color=colors[tide])
    ax.axvline(f_reg, lw=0.5, color='k', ls='--')
    ax.format(xreverse=False, yscale='log', xscale='log',
              xlim=(12/24, 12), ylim=(1.1e-8, 0.1), ylocator=10.**np.arange(-7, -1, 1),
              yminorlocator='log', xgrid=False,
              yformatter=['$10^{' + str(i) + '}$' for i in np.arange(-7, -1, 1)])

    ax.format(title=title, titlepad=6)
    
h = [ax.plot([], [], color='blue'), ax.plot([], [], color='red')]
l = ['CW', 'CCW']
for width, opacity in zip([1,3,5], [1, 0.5, 0.25]):
    h.append(ax.plot([],[], lw=width, alpha=opacity, color='k'))
l = l + ['Median', '25-75%', '10-90%']
l.append('Min/Max')
h.append(ax.plot([],[], color='k', ls=':'))
fig.legend(h, l, ncols=1, loc='r')
fig.save('../figures/figure4_abcd.png', dpi=300)

### Plot tide analysis box plots
fig, ax = pplt.subplots(ncols=2, sharey=False, sharex=False)

ax[0].boxplot(df_r2.loc[:, ['NB', 'YP', 'GC', 'GS']], cycle=['gold', 'tab:cyan', 'plum', 'lime8'])
ax[0].format(ylabel='$r^2$', xlabel='Region', ylim=(0,1), title='Coefficient of determination')

ax[1].boxplot(df_U.loc[:, ['NB', 'YP', 'GC', 'GS']], cycle=['gold', 'tab:cyan', 'plum', 'lime8'])
ax[1].format(ylabel='$U_{tide}$ (m/s)', xlabel='Region', ylim=(0,0.5), title='Maximum current',
            suptitle='All tidal constituents')

fig.save('../figures/figure4_fg.png', dpi=300)
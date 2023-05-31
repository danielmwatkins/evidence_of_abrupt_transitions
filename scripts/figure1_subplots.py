import matplotlib.patheffects as pe
import numpy as np
import os
import pandas as pd
import proplot as pplt
import pyproj
from scipy.interpolate import interp1d
import sys
import warnings
import xarray as xr

warnings.simplefilter("ignore")


#### Load buoy data ####
dataloc = '../data/daily_merged_buoy_data/'
dataloc_hourly = '../data/hourly_merged_buoy_data/'
files = os.listdir(dataloc)
buoy_data = {f.split('.')[0]: pd.read_csv(dataloc + f,
                 parse_dates=True, index_col=0) for f in files if f.split('.')[1] == 'csv'}
buoy_data_hourly = {f.split('.')[0]: pd.read_csv(dataloc_hourly + f,
                 parse_dates=True, index_col=0) for f in files if f.split('.')[1] == 'csv'}


for buoy in buoy_data_hourly:
    buoy_data_hourly[buoy]['speed_wind'] = (buoy_data_hourly[buoy]['u_wind']**2 + \
                                                buoy_data_hourly[buoy]['v_wind']**2)**0.5
    
extended_dn = ['2020P160', '2019P123', '2019P155', '2019P156',
               '2019P157', '2019P182', '2019P127', '2019P128',
                '2019P184']
dn = [b for b in buoy_data if b not in extended_dn]
dn = [b for b in dn if buoy_data[b].latitude.max() < 85]


depth = xr.open_dataset('../data/interpolated_depth.nc')

#### Plot the map #####
pplt.rc['reso'] = 'med'
plot_buoy = '2020P225'

colors = {m: c['color'] for m, c in zip(['May', 'June', 'July', 'August', 'September'],
                                        pplt.Cycle('spectral', N=5))}

fig, ax = pplt.subplots(proj='ortho', proj_kw={'lon_0': -5, 'lat_0': 75}, width=5)
ax.format(latlim=(70, 85), lonlim=(-25,10),  land=True,
          landcolor='light gray', latlabels=True, lonlabels=True)
cbar = ax.contourf(depth.longitude,
                depth.latitude,
                depth.z, levels=[-4000, -3500, -3000, -2500,
                              -2000, -1500, -1000, -500,
                              -200, -100, -50, 0],
                cmap='blues8_r',
                extend='both')

for buoy in extended_dn:
    ax.plot(buoy_data_hourly[buoy].longitude,
           buoy_data_hourly[buoy].latitude, color='gold', lw=1, alpha=0.75, zorder=2)

last_date = {}
for buoy in buoy_data:
    sic = buoy_data[buoy].sea_ice_concentration
    last = sic[sic > 1].index[-1]
    last_date[buoy] = last


for buoy in dn:
    buoy_df = buoy_data_hourly[buoy].resample('1H').asfreq()
    sic = buoy_data[buoy].sea_ice_concentration
    last = sic[sic > 1].index[-1]
    ts_ice = slice('2020-05-01', last)
    buoy_df = buoy_df.loc[ts_ice]

    ax.plot(buoy_df.longitude,
           buoy_df.latitude, color='r', lw=1, alpha=0.3, zorder=3)

    if buoy == plot_buoy:
        sic = buoy_data[buoy].sea_ice_concentration
        last = sic[sic > 1].index[-1]
        ts_ice = slice('2020-05-01', last)
        ts_water = slice(last + pd.to_timedelta('12H'), '2020-09-01 00:00')
        ax.plot(buoy_data_hourly[buoy].longitude.loc[ts_ice],
                buoy_data_hourly[buoy].latitude.loc[ts_ice],
                color='light gray', lw=2.5, zorder=5,
               path_effects=[pe.Stroke(linewidth=3.5, foreground='k'), pe.Normal()])
        ax.plot(buoy_data_hourly[buoy].longitude.loc[ts_water].resample('12H', offset=0).asfreq(),
                buoy_data_hourly[buoy].latitude.loc[ts_water].resample('12H', offset=0).asfreq(),
                color='light gray', lw=0, zorder=5, marker='.', edgecolor='k', edgewidth=0.5)
        buoy_data_hourly[buoy] = buoy_data_hourly[buoy].loc[ts_ice].copy()
        buoy_data[buoy] = buoy_data[buoy].loc[ts_ice].copy()        
        
for m, c in zip([5, 6, 7, 8], colors):
    date = pd.to_datetime('2020-' + str(m).zfill(2) + '-01 12:00')
    if date in buoy_data[plot_buoy].index:
        ax.plot(buoy_data[plot_buoy].loc[date, 'longitude'],
                buoy_data[plot_buoy].loc[date, 'latitude'], c=colors[c],
                marker='o', lw=0, edgecolor='k', s=5, zorder=6)

h = [ax.plot([],[], c=colors[c], marker='o', lw=0, edgecolor='k') for c in colors if c[0] != 'S']
l = [c[0:3] + ' 1st' for c in colors if c[0] != 'S']
ax.legend(h, l, ncols=1, loc='lr', pad=1, alpha=1)

h = [ax.plot([],[], c='light gray', lw=2.5,
            path_effects=[pe.Stroke(linewidth=3.5, foreground='k'), pe.Normal()]),
     ax.plot([],[],c='r', lw=2.5), ax.plot([],[], lw=2.5, color='gold')]
     
l = ['CO (' + plot_buoy + ')', 'DN (init<60 km)', 'Ext. DN (init>60 km)']
ax.legend(h, l, ncols=1, loc='ul', pad=1, alpha=1)


ax.colorbar(cbar, label='Depth (m)', loc='b')
fig.save('../figures/figure1_k.png', dpi=300)


#### Time Series ####
get_df = lambda varname: pd.DataFrame({b: buoy_data[b][varname] for b in dn})
get_df_hourly = lambda varname: pd.DataFrame({b: buoy_data_hourly[b][varname] for b in dn})
bliss_data = pd.read_csv("../data/MOSAiC_MIZ_hourly_mean_deformation_stdev.csv",
                        index_col=0, parse_dates=True)
df_sic = get_df('sea_ice_concentration')
speed_df = get_df('speed')
depth_df = get_df('depth')/1e3 # convert from meters to km

speed_anom = get_df_hourly('speed') - \
                    get_df_hourly('speed').rolling('24H', center=True).median()
wind_speed_df = get_df_hourly('speed_wind').resample('1D').median()


#### Key Figure (right hand side of it at least)
fig, axs = pplt.subplots(width=8, height=5, nrows=7,
                         sharey=False, hspace=[0,0,0,0,1,0])

timeslice = slice('2020-05-01', '2020-09-01')
ws_median = wind_speed_df[dn].resample('1D').median().loc[timeslice]
u_median = speed_df[dn].resample('1D').median().loc[timeslice]

### Panel a: Ensemble daily median drift speed
idx = 0
axs[idx].plot(u_median.median(axis=1),
        shadedata=[u_median.quantile(0.75, axis=1), u_median.quantile(0.25, axis=1)],
        fadedata=[u_median.quantile(0.90, axis=1), u_median.quantile(0.10, axis=1)],
         color='r', lw=1, shadealpha=0.35, fadealpha=0.2)

axs[idx].format(ylabel='$\overline{U}$ (m/s)',ylim=(0, 0.75), titlepad=1, ultitle='Sea ice median drift speed',
          titlecolor='k',
                yticks=[0.25, 0.5],
          xticks=[], ytickminor=False)

### Panel b: Sub-daily drift speed
idx += 1
axs[idx].plot(speed_anom.median(axis=1).loc[timeslice],
        shadedata=[speed_anom.quantile(0.75, axis=1).rolling('1D', center=True).median().loc[timeslice],
                   speed_anom.quantile(0.25, axis=1).rolling('1D', center=True).median().loc[timeslice],
                  ],
        fadedata=[speed_anom.quantile(0.90, axis=1).rolling('1D', center=True).median().loc[timeslice],
                  speed_anom.quantile(0.10, axis=1).rolling('1D', center=True).median().loc[timeslice],
                 ],
         color='r', lw=0.5, marker='', ms=1,shadealpha=0.35, fadealpha=0.2)

axs[idx].format(ylabel='U\' (m/s)',ylim=(-0.25, 0.25), ultitle='Sea ice drift speed anomaly',
          yticks=[-0.15, 0, 0.15], ytickminor=False,
          titlecolor='k',
          xticks=[])


### Panel c: Wind speed (ERA5)
idx += 1
axs[idx].plot(ws_median.median(axis=1),
        shadedata=[ws_median.quantile(0.75, axis=1), ws_median.quantile(0.25, axis=1)],
        fadedata=[ws_median.quantile(0.90, axis=1), ws_median.quantile(0.10, axis=1)],
         color='k', lw=1, shadealpha=0.25, fadealpha=0.15)
axs[idx].format(ylabel='$U_w$ (m/s)', ylim=(0,18),
                yticks=[5, 10],
                ultitle='ERA5 wind speed', titlecolor='k')


## Panel d: Sea ice concentration
idx += 1
axs[idx].plot(df_sic.median(axis=1).loc[timeslice],
        shadedata=[df_sic.quantile(0.75, axis=1).loc[timeslice],
                   df_sic.quantile(0.25, axis=1).loc[timeslice]],
        fadedata=[df_sic.quantile(0.9, axis=1).loc[timeslice],
                   df_sic.quantile(0.1, axis=1).loc[timeslice]],
         lw=1, shadealpha=0.25, fadealpha=0.15)
axs[idx].format(lltitle='Sea ice concentration', ylabel='SIC (%)',
                abcloc='ll', yticks=[15, 50, 100],
                ytickminor=False, ylim=(0, 110))

ax2 = axs[idx].twinx()
ax2.plot(df_sic.notnull().sum(axis=1), color='gray')
ax2.format(lrtitle='Number of buoys', ylabel='n', yticks=[10, 30, 50, 70], ytickminor=False, ylim=(0, 110))

h, l = [], []
for width, opacity in zip([1,3,5], [1, 0.5, 0.25]):
    h.append(ax.plot([],[], lw=width, alpha=opacity, color='k'))
l = ['Median', '25-75%', '10-90%']
axs[4].legend(h, l, ncols=1, loc='lr')

## Panel e: Depth
idx += 1

axs[idx].plot(depth_df.loc[:, dn].median(axis=1).loc[timeslice],
        shadedata=[depth_df.loc[:, dn].quantile(0.75, axis=1).loc[timeslice],
                   depth_df.loc[:, dn].quantile(0.25, axis=1).loc[timeslice]],
        fadedata=[depth_df.loc[:, dn].quantile(0.9, axis=1).loc[timeslice],
                   depth_df.loc[:, dn].quantile(0.1, axis=1).loc[timeslice]],
         color='brown',
         lw=1, shadealpha=0.15, fadealpha=0.1)
#axs[idx].plot(depth_df.loc[:, '2020P225'], color='brown', linestyle=':')
axs[idx].format(ylabel='Z (km)',
                ylocator=(-3, -1), ylim=(-4.5, 0),
                ytickminor=False, ultitle='Ocean depth')


## Panel f: Divergence
idx += 1
scale = 1e5
axs[idx].plot(bliss_data.loc[timeslice, 'divergence_area_weighted']*scale,
        shadedata=bliss_data.loc[timeslice, 'div_area_wt_stdev']*scale/np.sqrt(70),
        fadedata=bliss_data.loc[timeslice, 'div_area_wt_stdev']*scale,              
              shadealpha=0.35, fadealpha=0.2,
              color = 'purple', lw=0.75)
axs[idx].format(ylabel='$\epsilon_{I}$ (10$^5$ s$^{-1}$)',
                ylim=(-0.5, 0.5), ylocator=(-0.25, 0.25),
                ytickminor=False, ultitle='Divergence')
axs[idx].axhline(0, color='k', lw=0.75, linestyle=':')
axs.format(xlabel='', xrotation=45, xformatter='%b-%d',
          abc=True, abcloc='ul')
axs.format(titlepad=6, xlim=(pd.to_datetime('2020-05-01'), pd.to_datetime('2020-09-01 00:00')))

h, l = [], []
for width, opacity in zip([1,5], [1, 0.5]):
    h.append(ax.plot([],[], lw=width, alpha=opacity, color='purple'))
l = ['Area-weighted \nmean', 'Standard \ndeviation']
axs[-1].legend(h, l, loc='lr', ncols=1, alpha=1)


idx += 1
## Panel g:

axs[idx].plot(bliss_data.loc[timeslice, 'maximum_shear']*scale,
        shadedata=bliss_data.loc[timeslice, 'max_shear_stdev']*scale,
              color = 'purple', lw=0.75)

axs.format(xlabel='', xrotation=45, xformatter='%b-%d',
          abc=True, abcloc='ul')
axs.format(titlepad=6, xlim=(pd.to_datetime('2020-05-01'), pd.to_datetime('2020-09-01 00:00')))
axs[idx].format(ylabel='$\epsilon_{II}$ \n(10$^5$ s$^{-1}$)',
                ylim=(-0.1, 4), ylocator=(1, 3),
                ytickminor=False, ultitle='Maximum shear')



fig.save('../figures/figure1_abcdefg.png', dpi=300)

### Overlaying buoy positions on MODIS imagery ####

crs0 = pyproj.CRS('WGS84')
crs1 = pyproj.CRS('epsg:3413')
transformer_ll = pyproj.Transformer.from_crs(crs0, crs_to=crs1, always_xy=True)
transformer_xy = pyproj.Transformer.from_crs(crs1, crs_to=crs0, always_xy=True)

lats = np.arange(75, 90, 1)
lons = np.arange(-35, 30, 5)
lons, lats = np.meshgrid(lons, lats)
xylon, xylat = transformer_ll.transform(lons, lats)


x0 = 0.4e6
y0 = -1.1e6
lat_labels = []
lat_y = []
lon_labels = []
lon_x = []
for idx in range(0, xylon.shape[0]):
    if np.any(xylon[idx,:] < x0) & np.any(xylon[idx,:] > x0):
        y = interp1d(xylon[idx,:], xylat[idx,:])(x0)
        lat_y.append(y)
        lat_labels.append(lats[idx,0])
for idx in range(0, xylon.shape[1]):
    if np.any(xylat[:,idx] < y0) & np.any(xylat[:,idx] > y0):
        if np.any(xylon[:,idx] < x0) & np.any(xylon[:,idx] > x0):            
            x = interp1d(xylat[:,idx], xylon[:,idx])(y0)
            lon_x.append(x)
            lon_labels.append(lons[0,idx])
            
lat_y = list(np.array(lat_y))
lat_labels = [str(x) + '$^\circ$' for x in lat_labels]
lon_labels = [str(x) + '$^\circ$' for x in lon_labels]
lon_x = list(np.array(lon_x))
start = '2020-05-01 00:00' 
df_lon = pd.DataFrame({buoy: buoy_data[buoy]['longitude'].loc[slice(start, last_date[buoy])] for buoy in buoy_data})
df_lat = pd.DataFrame({buoy: buoy_data[buoy]['latitude'].loc[slice(start, last_date[buoy])] for buoy in buoy_data})
df_lon.index = df_lon.index - pd.to_timedelta('12H') # Set origin at 0 for plotting
df_lat.index = df_lat.index - pd.to_timedelta('12H')

pplt.rc['reso'] = 'med' 
files = ['2020-07-12T00_00_00Z.tiff',
         '2020-07-26T00_00_00Z.tiff',
         '2020-08-06T00_00_00Z.tiff']

fig, axs = pplt.subplots(width=7, ncols=3, share=False)
axs.format(grid=False)
for ax, file in zip(axs, files):
    date = pd.to_datetime(file.split('.')[0], format='%Y-%m-%dT%M_00_00Z')
    modis_image = xr.open_rasterio('../data/modis_imagery/' + file).sel(x=slice(2.9e5, 1.1e6),
                                                                        y=slice(-5e5, -1.2e6))
    X = modis_image.x.data
    Y = modis_image.y.data
    
    x, y = transformer_ll.transform(df_lon, df_lat)
    df_x = pd.DataFrame(x, columns=df_lon.columns, index=df_lon.index)
    df_y = pd.DataFrame(y, columns=df_lon.columns, index=df_lon.index)

    rgb = np.dstack([modis_image.sel(band=1).data, modis_image.sel(band=2).data, modis_image.sel(band=3).data])

    ax.imshow(rgb, extent=(X[0], X[-1], Y[-1], Y[0]))

    idx = df_x.loc[date,:] > X.min()
    idx = idx & (df_y.loc[date, :] > Y.min())
    ax.scatter(df_x.loc[date,idx], df_y.loc[date,idx], s=5, color='r')

    ex_dn = [b for b in df_x.columns if b in extended_dn]
    idx = df_x.loc[date,ex_dn] > X.min()
    idx = idx & (df_y.loc[date, ex_dn] > Y.min())
    ax.scatter(df_x.loc[date,ex_dn][idx], df_y.loc[date,ex_dn][idx], s=5, color='gold')


    ax.scatter(df_x.loc[date, '2020P225'], df_y.loc[date, '2020P225'], marker='*',
               c='light blue', edgecolor='k', s=150)
    
    for idx in range(xylon.shape[1]):
        ax.plot(xylon[:,idx], xylat[:,idx], color='light gray', lw=0.5)
    for idx in range(xylon.shape[0]):
        ax.plot(xylon[idx,:], xylat[idx,:], color='light gray', lw=0.5)

    ax.plot(xylon[0,:], xylat[0,:], color='r', lw=0.5)



    ax.format(title=date.strftime('%Y-%m-%d'), titlecolor='k', titlesize=13,
             xlim=(x0, 1e6),
             ylim=(-5e5, y0))

    ax.plot([x0+0.5e5, x0+1.5e5], [y0+5e4, y0+5e4], color='k', lw=2)
    ax.text(x0+0.4e5, y0+0.6e5, '100 km')

    ax.format(xtickminor=False,
              ytickminor=False,
             ylocator=lat_y,
             yformatter=lat_labels,
              xlocator=lon_x[2:],
              xformatter=lon_labels[2:],
              ylabel='', xlabel='')
    
axs.format(xreverse=False, yreverse=False)
fig.save('../figures/figure1_hij.png', dpi=300)
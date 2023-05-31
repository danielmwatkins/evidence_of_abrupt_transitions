import matplotlib.patheffects as pe
import metpy.calc as mcalc
from metpy.units import units
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
dataloc = '../data/hourly_merged_buoy_data/'
files = os.listdir(dataloc)
buoy_data = {f.split('.')[0]: pd.read_csv(dataloc + f,
                 parse_dates=True, index_col=0) for f in files if f.split('.')[1] == 'csv'}
    
extended_dn = ['2020P160', '2019P123', '2019P155', '2019P156',
               '2019P157', '2019P182', '2019P127', '2019P128',
                '2019P184']

ts = slice(pd.to_datetime('2020-05-01 00:00'), pd.to_datetime('2020-09-01 00:00'))
dn = [b for b in buoy_data if b not in extended_dn]
dn = [b for b in dn if buoy_data[b].latitude.max() < 85]

buoy_data = {buoy: buoy_data[buoy] for buoy in buoy_data if (buoy in dn) & (len(buoy_data[buoy].dropna()) > 24*30)}


#### Calculate drift speed ratio and drift speed offset
# Ratios are computed after taking the centered 12 hour median so that 
# inertial oscillations are removed

dfu = pd.DataFrame({b: buoy_data[b]['u'] for b in buoy_data}).rolling('12H',center=True).median().loc[ts]
dfv = pd.DataFrame({b: buoy_data[b]['v'] for b in buoy_data}).rolling('12H',center=True).median().loc[ts]
dfuw = pd.DataFrame({b: buoy_data[b]['u_wind'] for b in buoy_data}).rolling('12H',center=True).median().loc[ts]
dfvw = pd.DataFrame({b: buoy_data[b]['v_wind'] for b in buoy_data}).rolling('12H',center=True).median().loc[ts]

differences = {}
for buoy in dfu.columns:
    wind_bearing = mcalc.wind_direction(dfuw[buoy].values * units('m/s'),
                     dfvw[buoy].values * units('m/s'), convention='to')
    ice_bearing = mcalc.wind_direction(dfu[buoy].values * units('m/s'),
                     dfv[buoy].values * units('m/s'), convention='to')
    delta = np.deg2rad(ice_bearing.magnitude) - np.deg2rad(wind_bearing.magnitude)
    buoy_data[buoy]['turning_angle'] = pd.Series(np.rad2deg(np.arctan2(np.sin(delta), np.cos(delta))), index=dfv[buoy].index)

    wind_speed = (dfuw[buoy]**2 + dfvw[buoy]**2)**0.5
    ice_speed = (dfu[buoy]**2 + dfv[buoy]**2)**0.5
    buoy_data[buoy]['drift_speed_ratio'] = ice_speed / wind_speed
    
#### Estimate drift speed with wind model
get_df = lambda varname: pd.DataFrame({b: buoy_data[b][varname] for b in buoy_data})
ratios = get_df('drift_speed_ratio').resample('1D').median()
angles = get_df('turning_angle').resample('1D').median()

model_alpha = np.round(ratios.median(axis=0).median(), 3)
model_theta = np.round(angles.median(axis=0).median(), 3)

print('Alpha:', model_alpha)
print('Theta:', model_theta)

for buoy in buoy_data:
    u = dfu[buoy]
    v = dfv[buoy]
    uw = dfuw[buoy]
    vw = dfvw[buoy]
    U_est = model_alpha * np.exp(-1j*np.deg2rad(model_theta))*(uw + 1j*vw)
    u_est = pd.Series(np.real(U_est), index=u.index)
    v_est = pd.Series(np.imag(U_est), index=u.index)
    buoy_data[buoy]['u_est'] = u_est
    buoy_data[buoy]['v_est'] = v_est
    buoy_data[buoy]['speed_est'] = np.abs(U_est)

fig, axs = pplt.subplots(width=7, height=5, nrows=4,
                         sharey=False, hspace=0)
### Panel b: Ensemble drift speed ratio
idx = 0
u_est = get_df('speed_est').loc[ts]
u_median = get_df('speed').loc[ts]
resid = pd.DataFrame({buoy: u_median[buoy]- u_est[buoy]  for buoy in u_est.columns})

u_est = u_est.resample('1D').median()
u_median = u_median.resample('1D').median()
resid = resid.resample('1D').median()

axs[idx].plot(u_est.median(axis=1),     
        shadedata=[u_est.quantile(0.75, axis=1),
                   u_est.quantile(0.25, axis=1)],
        fadedata=[u_est.quantile(0.9, axis=1),
                   u_est.quantile(0.1, axis=1)],
        color='tab:blue'
 )
axs[idx].plot(u_median.median(axis=1),
        shadedata=[u_median.quantile(0.75, axis=1), u_median.quantile(0.25, axis=1)],
        fadedata=[u_median.quantile(0.90, axis=1), u_median.quantile(0.10, axis=1)],
         color='r', lw=1, shadealpha=0.15, fadealpha=0.1)
axs[idx].format(ylabel='$\overline{U}$ (m/s)',ylim=(0, 0.75), titlepad=1, ultitle='Sea ice median drift speed',
          titlecolor='k',
          xticks=[], ytickminor=False)


idx += 1

axs[idx].plot(resid.median(axis=1),     
        shadedata=[resid.quantile(0.75, axis=1),
                   resid.quantile(0.25, axis=1)],
        fadedata=[resid.quantile(0.9, axis=1),
                   resid.quantile(0.1, axis=1)],
        color='tab:gray'
 )
axs[idx].format(ylabel='$\overline{U} - \overline{U}_{est}$ (m/s)',ylim=(-0.35, 0.35), titlepad=1,
                ultitle='Error in drift speed magnitude',
          titlecolor='k',
          xticks=[], ytickminor=False)
axs[idx].axhline(0, color='k', lw=0.5)
idx += 1
axs[idx].plot(ratios.median(axis=1),
        shadedata=[ratios.quantile(0.75, axis=1),
                  ratios.quantile(0.25, axis=1)],
        fadedata=[ratios.quantile(0.9, axis=1),
                  ratios.quantile(0.1, axis=1)],        
        label='$\left \langle \\frac{\overline{u}}{\overline{u_{wind}}} \\right \\rangle_{DN}$',
        color='green', lw=1, shadealpha=0.15, fadealpha=0.1)
axs[idx].format(ultitle='Drift speed ratio',
          ylabel='$\\alpha$', xticks=[], yticks=[0.025, 0.05, 0.075, 0.1],
          yformatter=['', '5%', '', '10%'], ytickminor=False, ylim=(0, 0.125))
axs[idx].axhline(0.02, ls='--', color='k', lw=1)
axs[idx].text(pd.to_datetime('2020-05-05 00:00'), 0.025, '$\\alpha = 2$%')

### Panel c: Drift offset
idx += 1
axs[idx].plot(angles.resample('1D').median().median(axis=1),
        shadedata=[angles.resample('1D').median().quantile(0.75, axis=1),
                  angles.resample('1D').median().quantile(0.25, axis=1)],
        fadedata=[angles.resample('1D').median().quantile(0.9, axis=1),
                  angles.resample('1D').median().quantile(0.1, axis=1)],
        color='mauve',  lw=1, shadealpha=0.15, fadealpha=0.1) 

axs[idx].axhline(20.5, color='k', lw=1, ls='--')
axs[idx].text(pd.to_datetime('2020-05-05 00:00'), 30, '$\\theta = 20.5^\circ$')

axs[idx].axhline(0, color='gray', lw=1, zorder=0)
axs[idx].format(ultitle='Turning angle', ytickminor=False,
          ylabel='$\\theta$', yticks=[-90, -45, 0, 45, 90], ylim=(-135, 135))

h, l = [], []
for color in ['r', 'tab:blue'] :
    h.append(axs[0].plot([],[], lw=2, color=color))
l = ['Observed', 'Estimated']
axs[0].legend(h, l, ncols=3, loc='uc')

h, l = [], []
for width, opacity in zip([1,3,5], [1, 0.5, 0.25]):
    h.append(axs[0].plot([],[], lw=width, alpha=opacity, color='k'))
l = ['Median', '25-75%', '10-90%']
axs[1].legend(h, l, ncols=3, loc='ll')

axs.format(xlabel='', xrotation=45, xformatter='%b-%d',
          abc=True, abcloc='ul')
axs.format(titlepad=6)

fig.save('../figures/figure2_abcd.jpg', dpi=300)

### Histograms of drift speed ratio and turning angle
fig, axs = pplt.subplots(nrows=2, share=False, height=6)
ax = axs[0]
ts0 = slice('2020-06-16', '2020-07-14')
ts1 = slice('2020-07-15', '2020-08-14')

alpha_median1 = ratios.loc[ts0].melt().value.dropna().median()
x = ax.hist(ratios.loc[ts0].melt().value.dropna(),
            bins=np.linspace(0, 0.2, 50), density=True,
            alpha=0.5, c='g')
alpha_median2 = ratios.loc[ts1].melt().value.dropna().median()
x = ax.hist(ratios.loc[ts1].melt().value.dropna(),
            bins=np.linspace(0, 0.2, 50), density=True,            
        histtype='step', lw=1, alpha=1, c='g')
ax.format(title='', xlabel='Drift speed ratio')

ax = axs[1]

theta_median1 = angles.loc[ts0].melt().value.dropna().median()
x = ax.hist(angles.loc[ts0].melt().value.dropna(),
            bins=np.linspace(-180, 180, 50),
            density=True,
            alpha=0.5, c='mauve')
theta_median2 = angles.loc[ts1].melt().value.dropna().median()
x = ax.hist(angles.loc[ts1].melt().value.dropna(),
            bins=np.linspace(-180, 180, 50), density=True,
            histtype='step', lw=1, alpha=1, c='mauve')


ax.format(title='', xlabel='Turning angle',
          xlim=(-180,180))

# Legend
h = [ax.plot([],[], lw=0, marker='s', color='gray', markersize=8),
    ax.plot([],[], lw=0, marker='s', edgecolor='gray', color='w', markersize=8)]
l = ['Jun. 15-Jul. 14 \n$\\alpha_{0.5} = $' +  str(np.round(alpha_median1*100, 1)) + '%' + \
     '\n$\\theta_{0.5} = $' +  str(np.round(theta_median1, 1)) + '$^\circ$',
     'Jul. 15-Aug. 14\n$\\alpha_{0.5} = $' +  str(np.round(alpha_median2*100, 1)) + '%' + \
     '\n$\\theta_{0.5} = $' +  str(np.round(theta_median2, 1)) + '$^\circ$']

fig.legend(h, l, loc='b', ncols=2)

fig.save('../figures/figure2_ef.jpg', dpi=300)
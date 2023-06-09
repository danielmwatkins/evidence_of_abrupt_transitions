"""Applies the harmonic analysis from Pease et al to the time series used in the rotary spectra analysis"""
import numpy as np
import os
import pandas as pd
import pyproj
import sys
from drifter import compute_velocity

### Parameters
tc = pd.Series(
    [0.92954, 0.99726, 1.,  1.00274, 1.89598, 1.93227, 2.0],
    index=['O1', 'P1', 'S1', 'K1', 'N2', 'M2', 'S2']
) # Units are cycles per day

tidal_constituents = pd.DataFrame({'idx': np.arange(1, len(tc)+1),
                                   'cpd': tc,
                                   'cps': tc * 2*np.pi/(24*3600)})

# Low pass filter - units are indices, so for this data it should be hours. Should match what is used in the spectra calc
lpfilter = 35
min_length = '20D'

### Function definitions
def bathymetric_regions(buoy, buoy_data):
    """Returns bathymetric region based on the following criteria:
    Nansen Basin: deeper than 3000 m, north of 82, south of 85
    Yermak Plateau: Between 80 and 84 latitude, east of 0 longitude, shallower than 1000 m
    E. GL. Channel: Between 75 and 82.5 latitude, west of 2 longitude, deeper than 1500 m
    E. GL. Shelf: Between 65 and 83 latitude, between -25 and 0 longitude, shallower than 500 m."""
    
    lat = buoy_data[buoy]['latitude']
    lon = buoy_data[buoy]['longitude']
    depth = buoy_data[buoy]['depth']
    
    nansen = ((lat > 82) & (lon > 0)) & ((lat < 85) & (depth < -3000))
    yermak = ((lat < 84) & (lon > 0)) & ((lat > 80) & (depth > -1000))
    channel = ((lat < 82.5) & (lat > 75)) & ((lon < 2) & (depth < -1500))
    shelf = (((lat < 83) & (lat > 65)) & ((lon < 0) & (lon > -25)) & (depth > -500))
    
    channel = channel & ~yermak
    return {'NB': nansen, 'YP': yermak, 'GC': channel, 'GS': shelf}



# Coriolis frequency as a function of latitude
f = lambda latitude: 2*2*np.pi/(24*3600)*np.sin(np.deg2rad(latitude))
fperday = lambda latitude: 24 * 3600 * f(latitude) / (2*np.pi)
from scipy.optimize import minimize

def build_z(t, tc=None, latitude=None, inertial=True):
    """Make the Z matrix for fitting the harmonic model. 
    t is in seconds, tc is either none or a list of constituents. If inertial
    oscillations are to be calculated, must supply values for latitude. Fails
    if both tc=None and inertial=False."""

    n = len(t)
    if tc is not None:
        Z11 = np.vstack([[np.ones(n)] + \
               [t] + \
               [np.cos(w*t) for w in tc['cps']] + \
               [np.sin(w*t) for w in tc['cps']]
              ]).T
    else:
        Z11 = np.vstack([[np.ones(n)] + [t]]).T

    if inertial:
        Z12 = np.vstack([np.cos(f(latitude) * t),
                     np.sin(f(latitude) * t),
                    ]).T
    
        k = Z11.shape[1] + Z12.shape[1]
        Z00 = np.zeros((n, k))
        Z = np.vstack([np.hstack([Z11, Z12, Z00]),
                   np.hstack([Z00, Z11, Z12])])

    else:
        k = Z11.shape[1]
        Z00 = np.zeros((n, k))
        Z = np.vstack([np.hstack([Z11, Z00]),
                      np.hstack([Z00, Z11])])
    return Z

def fit_harmonic_model(df_sel, xvar='x', yvar='y', 
                          tidal_constituents=None, inertial=True,
                          max_iterations=100,
                          reference_time=pd.to_datetime('2020-05-01 00:00')):
    """Use sequential least squares to find a best fit harmonic approximation.
    Follows main details of Pease et al. 1995 and Turet et a. 1993. The dataframe
    df_sel should have x and y anomalies as columns labeled <xvar> and <yvar>
    and should have a latitude variable. Tidal constituents be a dataframe
    with idx for the numbering of coefficients and cps for the frequency with units of s^-1. 
    
    To calculate tide influence without inertial oscillations, set inertial=False
    and supply tidal_constituent frequencies (rad/sec).
    Return constituents. TBD: Depart from Pease naming convention and use the tidal constituents
    instead of numbers, this is simpler.
    """
    # Build matrix

    n = len(df_sel)
    # need to address the nan's (TBD)

    t = (df_sel.index - reference_time).total_seconds()
    Z = build_z(t,
            tc=tidal_constituents,
            latitude=df_sel['latitude'],
            inertial=inertial)
    k = 2
    if tidal_constituents is not None:
        k += len(tidal_constituents)
        
    if inertial:
        k += 2
        
    Y = np.hstack([df_sel[xvar], df_sel[yvar]])

    # Variable names to match Pease et al. formula
    v = ['C', 'D']
    if tidal_constituents is not None:
        for i in tidal_constituents.index:
            v.append('A' + str(i))
            v.append('B' + str(i))
    if inertial:
        v.append('E')
        v.append('F')
    v = [vv + 'x' for vv in v] + \
           [vv + 'y' for vv in v]

    # set up condition matrix so that A*x = 0
    # to force the inertial oscillations to be counterclockwise
    A = np.zeros((2, 2*k))
    
    if inertial:
        ii, jj = np.argwhere(np.array([vv[0] for vv in v]) == 'E').squeeze()
        A[0, jj] = -1
        A[1, ii] = 1

        ii, jj = np.argwhere(np.array([vv[0] for vv in v]) == 'F').squeeze()        
        A[0, ii] = 1
        A[1, jj] = 1

    def min_fun(x):
        return np.linalg.norm(np.abs(np.matmul(Z, x) - Y)) # Could do a norm instead of max

    def constraint_fun(x):
        return np.linalg.norm(np.abs(np.matmul(A, x)))

    # Initialize from random vector
    # Works OK for tides, but not for inertial oscillations
    xscale = np.diff(np.quantile(np.abs(Y),[0.25, 0.75]))
    x0 = np.random.normal(size=len(v))*xscale
    
    # Find best fit
    if inertial:
        y0 = minimize(min_fun,
                  x0,
                  tol=1e-16,
                  method='SLSQP',
                    constraints={
                        'type': 'eq',
                        'fun': constraint_fun
                    },
                  options={
                      'maxiter': max_iterations
                  })
    else:
        y0 = minimize(min_fun,
          x0,
          tol=1e-16,
          method='SLSQP',
          options={
              'maxiter': max_iterations
          })
        
    beta = pd.Series(y0['x'], index=v)
    n = int(Z.shape[0]/2)
    Y = np.matmul(Z, beta.values)
    x_fit = Y[0:n]
    y_fit = Y[n:]
    return beta, x_fit, y_fit, Z

#### Read in data and separate into bathymetric regions
dataloc = '../data/hourly_merged_buoy_data/'
buoy_data = {file.split('.')[0]:
                 pd.read_csv(dataloc + file,
                              index_col=0, parse_dates=True)
             for file in os.listdir(dataloc) if file.split('.')[-1] == 'csv'}

# Add polar stereographic coordinates
projIn = 'epsg:4326' # WGS 84 Ellipsoid
projOut = 'epsg:3413' # NSIDC North Polar Stereographic
transformer = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)

for buoy in buoy_data:
   
        lon = buoy_data[buoy].longitude.values
        lat = buoy_data[buoy].latitude.values

        x, y = transformer.transform(lon, lat)
        buoy_data[buoy]['x'] = x
        buoy_data[buoy]['y'] = y
        
buoy_subsets = {'NB': {},
                'YP': {},
                'GC': {},
                'GS': {}
               }

dt = pd.to_timedelta(min_length)
min_obs = 24*18 # At least 20 days of data
for buoy in buoy_data:
    regions = bathymetric_regions(buoy, buoy_data)
    for reg in regions:
        if np.sum(regions[reg]) > min_obs:
            idx = regions[reg]
            begin = idx[idx].index[0]
            end = begin + dt
            buoy_df = buoy_data[buoy].loc[slice(begin, end)].copy()            
            ref_index = pd.Series(np.nan, index=pd.date_range(begin, end, freq='1H'), name='new_index')
            if (len(buoy_df) != len(ref_index) & len(buoy_df)) > (0.95*len(ref_index)):
                # Note: for the "hourly_merged" dataset, it's already at hourly resolution
                # so this doesn't get used.
                buoy_df = buoy_df.merge(new_index, left_index=True, right_index=True)
                buoy_df = buoy_df.interpolate().fillna(0)

            if len(buoy_df) == len(ref_index):
                buoy_subsets[reg][buoy] = buoy_df
                
# For greenland channel, remove tracks that stay out of the center of the channel
buoy_subsets['GC'] = {b: buoy_subsets['GC'][b] for b in buoy_subsets['GC'] if \
                        buoy_subsets['GC'][b]['depth'].max() < -1000}            


ellipse_params = {region: {} for region in buoy_subsets}
tc_set = ['K1','O1', 'S2', 'M2', 'N2']
for region in buoy_subsets:
    for buoy in buoy_subsets[region]:
        buoy_df = buoy_subsets[region][buoy].copy()
        columns = buoy_df.columns
        
        buoy_df['x_anom'] = buoy_df['x'] - buoy_df['x'].rolling(
            lpfilter, center=True, win_type='bartlett').mean()
        buoy_df['y_anom'] = buoy_df['y'] - buoy_df['y'].rolling(
            lpfilter, center=True, win_type='bartlett').mean()
        buoy_subsets[region][buoy]['x_anom'] = buoy_df['x_anom']
        buoy_subsets[region][buoy]['y_anom'] = buoy_df['y_anom']
        
        buoy_df = buoy_df.dropna(subset='x_anom').copy()

        beta, x_fit, y_fit, Z = fit_harmonic_model(
                                        buoy_df,
                                        tidal_constituents=tidal_constituents.loc[tc_set, :],
                                        xvar='x_anom', yvar='y_anom', inertial=False)
        if 'A1x' not in buoy_df.columns:
            for v in beta.index:
                buoy_df[v] = 0 

        for v in beta.index:
            buoy_df.loc[buoy_df.index, v] = beta[v]

        idx = buoy_df.index # This allows the nan's present in the initial array to be skipped
        buoy_subsets[region][buoy]['x_tide'] = np.nan
        buoy_subsets[region][buoy]['y_tide'] = np.nan        
        buoy_subsets[region][buoy].loc[idx, 'x_tide'] = x_fit
        buoy_subsets[region][buoy].loc[idx, 'y_tide'] = y_fit
        buoy_subsets[region][buoy]['x_notide'] = buoy_subsets[region][buoy]['x_anom'] - \
                                                    buoy_subsets[region][buoy]['x_tide']
        buoy_subsets[region][buoy]['y_notide'] = buoy_subsets[region][buoy]['y_anom'] - \
                                                    buoy_subsets[region][buoy]['y_tide']


        temp = compute_velocity(buoy_subsets[region][buoy].drop(['x', 'y'], axis=1).rename(
                                         {'x_tide': 'x',
                                          'y_tide': 'y'}, axis=1),
                               date_index=True, rotate_uv=False, method='centered')
        buoy_subsets[region][buoy]['u_tides'] = temp['u']
        buoy_subsets[region][buoy]['v_tides'] = temp['v']


results = []
for region in buoy_subsets:
    for buoy in buoy_subsets[region]:
        buoy_df = buoy_subsets[region][buoy].copy()

        x = buoy_df['x_anom']
        xf = buoy_df['x_tide']
        y = buoy_df['y_anom']
        yf = buoy_df['y_tide']
        r2 = 1 - np.sum((x - xf)**2 + (y - yf)**2)/np.sum((x - x.mean())**2 + (y - y.mean())**2)
        umax = np.sqrt(buoy_df['u_tides']**2 + buoy_df['v_tides']**2).resample('1D').max().median()
        results.append([region, buoy, r2, umax])
        
        buoy_df.to_csv('../data/harmonic_fit/' + region + '/' + buoy + '_harmonic_fit.csv')
        
df_r2 = pd.DataFrame(results, columns=['region', 'buoy', 'r2', 'U']).pivot_table(index='buoy', values='r2', columns='region')
df_U = pd.DataFrame(results, columns=['region', 'buoy', 'r2', 'U']).pivot_table(index='buoy', values='U', columns='region')
df_r2.to_csv('../data/harmonic_fit/squared_correlation_coefficients.csv')
df_U.to_csv('../data/harmonic_fit/maximum_daily_current.csv')

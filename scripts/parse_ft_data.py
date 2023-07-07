"""Code to parse the IFT matlab files and produce CSV files that have all the information in place.

The parse_raw requires the data in <dataloc> to have a folder for each year. In each folder, there is an <input> and a <tracked> folder. The <tracked> folder contains order.mat, delta_t.mat, x3.mat, and y3.mat.

I used the data from Rosalinda L-A's folder with the path UCR/Research/ice_tracker/server/single_disp/data/disp_<year>/output/tracked

The time data file is produced by merging SOIT output with a file linking the day of year and the original file name, 
which contained the information on which satellite the image comes from.
"""
import numpy as np
import pandas as pd
import pyproj 
import rasterio
from scipy.interpolate import interp1d
from scipy.io import loadmat

saveloc = '../data/floe_tracker/parsed/'
dataloc = '../data/floe_tracker/unparsed/' 

for year in range(2003, 2021):
    X1 = loadmat(dataloc + str(year) + '/tracked/x3.mat')['x2']
    Y1 = loadmat(dataloc + str(year) + '/tracked/y3.mat')['y2']
#     order = loadmat(dataloc + str(year) + '/tracked/order.mat')['order'].squeeze()
#     dt = np.hstack([0, loadmat(dataloc + str(year) + '/tracked/delta_t.mat')['delta_t'].squeeze()])
#     dt = pd.to_timedelta(dt, unit='min')
#     dt_elapsed = np.cumsum(dt)

    info_df = pd.read_csv('{d}/{y}/time_data.csv'.format(
        d=dataloc, y=year), index_col=0)
    info_df['datetime'] = pd.to_datetime(info_df['SOIT time'])
    info_df.set_index('matlab_index', inplace=True)

    # Convert from pixels to stereographic coordinates
    # For 2003-2019 the reference file is the NE Greenland one
    # For 2020 it's the Fram Strait one. 
    info_region_pixel_scale_x = 256
    info_region_pixel_scale_y = 256
    x_cropped = 2.4745e+05
    y_cropped = -6.3589e+05

    if year == 2020:
        info_region_pixel_scale_x = 200.2979
        info_region_pixel_scale_y = 208.3310
        x_cropped = 2.0080e+05
        y_cropped = -3.1754e+05

#     if np.all(order == np.sort(order)):
#         df_x = pd.DataFrame(X1)
#         df_y = pd.DataFrame(Y1)
        
#     else:
#         df_x = pd.DataFrame(X1, columns=order)
#         df_y = pd.DataFrame(Y1, columns=order)        
#         df_x = df_x.loc[:, np.sort(order)]
#         df_y = df_y.loc[:, np.sort(order)]
        
#         info_df = info_df.sort_index()

#     df_x.columns = info_df['SOIT time']
#     df_y.columns = info_df['SOIT time']
    df_x = pd.DataFrame(X1, columns=info_df['SOIT time'])
    df_x.columns.name = 'datetime'
    df_x.index.name = 'floe_idx'
    df_x = df_x.where(df_x != 0)

    df_y = pd.DataFrame(Y1, columns=info_df['SOIT time'])
    df_y.columns.name = 'datetime'
    df_y.index.name = 'floe_idx'
    df_y = df_y.where(df_y != 0)

    df_x = x_cropped + df_x * info_region_pixel_scale_x
    df_y = y_cropped - df_y * info_region_pixel_scale_y

    pol_stere_projection = 'epsg:3413'
    crs0 = pyproj.CRS('WGS84')
    crs1 = pyproj.CRS(pol_stere_projection)
    transformer = pyproj.Transformer.from_crs(crs1, crs_to=crs0, always_xy=True)
    count = 0
    long_enough = []
    floe_dfs = {}
    for idx in df_x.index:
        df_idx = pd.DataFrame({'x': df_x.loc[idx, :].dropna(),
                                 'y': df_y.loc[idx, :].dropna()})

        x = df_idx['x']
        y = df_idx['y']
        lon, lat = transformer.transform(x, y)
        df_idx['latitude'] = np.round(lat, 5)
        df_idx['longitude'] = np.round(lon, 5)

        if len(df_idx) > 3:
            floe_dfs[idx] = df_idx
    df_xy = pd.concat(floe_dfs)
    
    # Sort the dates
    df_xy = df_xy.T.sort_index().T
    
    df_xy = df_xy.reset_index()
    df_xy.rename({'level_0': 'floe_id', 'level_1': 'datetime'}, axis=1, inplace=True)
    
    
    if year == 2020:
        # The images from 2020 are stretched in the y direction. This applies a linear correction.
        left=200703.99999999994
        bottom=-2009088.0
        right=1093632.0
        top=-317440.0
        adjustment = 63.8e3
        A = ((top - bottom) + adjustment)/(top - bottom)
        B = top * (1 - A)
        df_xy['y'] = A*df_xy['y'] + B
        source_crs = 'epsg:3413'
        to_crs = 'WGS84'
        ps2ll = pyproj.Transformer.from_crs(source_crs, to_crs, always_xy=True)
        lon, lat = ps2ll.transform(df_xy['x'], df_xy['y'])

        df_xy['longitude'] = np.round(lon, 5)
        df_xy['latitude'] = np.round(lat, 5)
    
    df_xy['x'] = np.round(df_xy['x'], 1)
    df_xy['y'] = np.round(df_xy['y'], 1)
    df_xy.to_csv(saveloc + 'floe_tracker_raw_' + str(year) + '.csv', index=False)
    
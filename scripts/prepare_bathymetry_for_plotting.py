import xarray as xr
import pyproj
import numpy as np
ds_depth = xr.open_dataset('../external_data/IBCAO_v4_2_400m.nc')
lon = np.arange(-80, 80, 0.5)
lat = np.arange(55, 88, 0.25)
lon_grid, lat_grid = np.meshgrid(lon, lat)
x_grid, y_grid = np.zeros(lon_grid.shape), np.zeros(lat_grid.shape)

# To plot the depth contours, I need to get the locations on a lat/lon grid
z_grid = np.zeros(x_grid.shape)

# Transform lon/lat grid to x/y
crs0 = pyproj.CRS('WGS84')
crs1 = pyproj.CRS('epsg:3996') # IBCAO Polar Stereographic

transformer = pyproj.Transformer.from_crs(crs0, crs_to=crs1, always_xy=True)
x_grid, y_grid = transformer.transform(lon_grid, lat_grid)
ds_sel = ds_depth.sel(x=slice(x_grid.min(), x_grid.max()),
                      y=slice(y_grid.min(), y_grid.max()))

x = np.array(ds_sel.x.data).astype(float)
y = np.array(ds_sel.y.data).astype(float)       

x_grid = xr.DataArray(x_grid, dims=('latitude', 'longitude'))
y_grid = xr.DataArray(y_grid, dims=('latitude', 'longitude'))

z_interp = ds_sel['z'].astype(np.float64).interp({'x': x_grid, 'y': y_grid}, method='linear')
ds  = xr.Dataset({'z': (('latitude', 'longitude'), z_interp.data),
           'latitude': (('latitude',), lat),
           'longitude': (('longitude',), lon)})

ds.to_netcdf('../data/interpolated_depth.nc')
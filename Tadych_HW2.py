# %% Importing relevant packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset 
import cartopy 
import cartopy.feature as cfeature
import cartopy.crs as ccrs

# %%
## Opening the data with xarray
data = xr.open_dataset('Data/merg_2018101012_4km-pixel.nc4')
data

value = data.Tb[0] # first time
lon = data["lon"]
lat = data["lat"]

# %% Looking at the metadata
metadata = data.attrs
metadata

# %% Slicing the dat
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
time = data.variables['time'][:]
Temp = data['Tb']
print(Temp)

# %% Quick Autoplot
Temp[0,:,:].plot()

# %% Bounding the area of interest
lon_min = -67
lon_max = -97
lat_min = 20
lat_max = 40
# print(lon_min)

# %% Plot it guuuurl
# matplotlib inline
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection=ccrs.PlateCarree())

#ax = plt.subplot(2, 2, 1)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, alpha = 1.0)
ax.set_title('Hurricane Michael (IMERG IR 2018-10-10 12:00 UTC)', fontsize = 14)
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

# Grid and Labels
gl = ax.gridlines(crs=crs, draw_labels=True, alpha=0.5)
gl.xlabels_top = None
gl.ylabels_right = None
xgrid = np.arange(lon_min-0.5, lon_max+0.5, 1.)
ygrid = np.arange(lat_min, lat_max+1, 1.)

# color bar
color = ax.pcolormesh(lon, lat, value, cmap = 'cool')

cbar = fig.colorbar(color, ax = ax, 
                orientation = 'horizontal', 
                fraction = 0.03, 
                # anchor = 0,
                pad = 0.04)
cbar.set_label("Temp (K)", fontsize = 12)
cbar.ax.tick_params(labelsize = 12)
plt.show()
plt.savefig("Tadych_Assignment2_HurricaneMichael")

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
# %%
# matplotlib inline
fig = plt.figure(figsize=(40,30))
ax = plt.axes(projection=ccrs.PlateCarree())

#ax = plt.subplot(2, 2, 1)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, alpha = 1.0)
ax.set_title('IMERG IR 2022-08-20 00:00 UTC', fontsize = 44)

# color bar
color = ax.pcolormesh(lon, lat, value, cmap = 'jet')

cbar = fig.colorbar(color, ax = ax, orientation = 'horizontal', fraction = 0.03, pad = 0.02)
cbar.set_label("Temp (K)", fontsize = 28)
cbar.ax.tick_params(labelsize = 24)
plt.show()
# %%

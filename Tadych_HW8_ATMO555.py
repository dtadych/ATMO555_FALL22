# %%
import requests
import netCDF4
import numpy as np
import xarray as xr

# %% Importing relevant packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset 
import cartopy 
import cartopy.feature as cfeature
import cartopy.crs as ccrs

# %%
datadirectory = 'Data/GPM_IMERG/'

# %% gotta make a list of names
Early = [
        # "3B-DAY-E.MS.MRG.3IMERG.20170823-S000000-E235959.V06.nc4",
        "3B-DAY-E.MS.MRG.3IMERG.20170824-S000000-E235959.V06.nc4",
        "3B-DAY-E.MS.MRG.3IMERG.20170825-S000000-E235959.V06.nc4",
        "3B-DAY-E.MS.MRG.3IMERG.20170826-S000000-E235959.V06.nc4",
        "3B-DAY-E.MS.MRG.3IMERG.20170827-S000000-E235959.V06.nc4"]
Late = ["3B-DAY-L.MS.MRG.3IMERG.20170823-S000000-E235959.V06.nc4",
        "3B-DAY-L.MS.MRG.3IMERG.20170824-S000000-E235959.V06.nc4",
        "3B-DAY-L.MS.MRG.3IMERG.20170825-S000000-E235959.V06.nc4",
        "3B-DAY-L.MS.MRG.3IMERG.20170826-S000000-E235959.V06.nc4",
        "3B-DAY-L.MS.MRG.3IMERG.20170827-S000000-E235959.V06.nc4"]
Total = ["3B-DAY.MS.MRG.3IMERG.20170823-S000000-E235959.V06.nc4",
        "3B-DAY.MS.MRG.3IMERG.20170824-S000000-E235959.V06.nc4",
        "3B-DAY.MS.MRG.3IMERG.20170825-S000000-E235959.V06.nc4",
        "3B-DAY.MS.MRG.3IMERG.20170826-S000000-E235959.V06.nc4",
        "3B-DAY.MS.MRG.3IMERG.20170827-S000000-E235959.V06.nc4"]
# %% --- Reading in the data ---
data_early = xr.open_dataset(datadirectory+"3B-DAY-E.MS.MRG.3IMERG.20170823-S000000-E235959.V06.nc4")
data_early
# %% This makes them all come together
for i in Early:
    f = xr.open_dataset(datadirectory+i)
    data_early = xr.merge([data_early,f])

print(data_early)
# %% Slicing the dat
data = data_early
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
time = data.variables['time'][:]
earlyprecip = data['precipitationCal']
print(earlyprecip)
# %%
global_mean = earlyprecip.sum(("time"))

# %%
global_mean.plot()

#%%
# Bounding the area of interest
lon_min = -67
lon_max = -97
lat_min = 20
lat_max = 40
# print(lon_min)
# global_mean[global_mean["lat"] >= lat_min].plot()

fig, ax = plt.subplots(figsize = (10,6))
# fig = plt.figure(figsize=(10,10))
# ax = plt.axes(projection=ccrs.PlateCarree())

xlim = (lat_min,  lat_max)
ylim = (lon_min,  lon_max)

ax.set_xlim(xlim)
ax.set_ylim(ylim)

global_mean.plot(ax = ax)
# ax.add_feature(cfeature.COASTLINE)
ax.set_title('Hurricane Harvey \n(IMERG Total Early for 8/23-8/27)', fontsize = 14)
ax.set_axis_off()
plt.show()

# %% Plot it guuuurl

value = data.precipitationCal[0] # first time
lon = global_mean["lon"]
lat = global_mean["lat"]

# matplotlib inline
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection=ccrs.PlateCarree())

#ax = plt.subplot(2, 2, 1)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, alpha = 1.0)
ax.set_title('Hurricane Harvey (IMERG IR 2017-10-10 12:00 UTC)', fontsize = 14)
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

# Grid and Labels
# gl = ax.gridlines(crs=crs, draw_labels=True, alpha=0.5)
# gl.xlabels_top = None
# gl.ylabels_right = None
# xgrid = np.arange(lon_min-0.5, lon_max+0.5, 1.)
# ygrid = np.arange(lat_min, lat_max+1, 1.)

# color bar
color = ax.pcolormesh(lon, lat, value, cmap = 'cool')

cbar = fig.colorbar(color, ax = ax, 
                orientation = 'horizontal', 
                fraction = 0.03, 
                # anchor = 0,
                pad = 0.04)
cbar.set_label("precipitationCal", fontsize = 12)
cbar.ax.tick_params(labelsize = 12)
plt.show()
# plt.savefig("Tadych_Assignment8_HurricaneHarvey")

# %%
f, ax = plt.subplots(figsize=(8, 5))
ax.plot(global_mean, color='#2F2F2F', label='Arizona Average')
ax.set(title='Global Average Precipitation for 1984')
# ax.legend()
ax.set_xlim(0,12)
ax.grid(zorder = 0)
plt.xlabel('Month')
plt.ylabel('Precipitation (mm/day)')
# %%

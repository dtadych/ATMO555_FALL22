#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:12:32 2022

@author: samdahl
"""

# importing the modules
import numpy as np
import matplotlib.pyplot as plt
import xarray 
from netCDF4 import Dataset 
import cartopy 
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from shapely import speedups

speedups.disable()


# opening the data
data = xarray.open_dataset('/Users/samdahl/Desktop/ATMO_455/CER_GEO_Ed4_GOE16_NH_V01.2_2021.152.2330.06K.nc')

# Calling the variables
T67 = data['temperature_67']-273 # Channel 67
T11 = data['temperature_ir']-273 # Channel 11
lat = data.coords['latitude'] # latitude 
lon = data.coords['longitude'] # longitude
T67

# plotting the base map with features 
figure = plt.figure(figsize=(24,24)) # base figure
ax = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree()) #IR Axis 

# plotting each of the features
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, alpha = 1.0)
ax.set_title('GOES EAST T11 (Infrared)', fontsize = 22)

# plot channel 6.7

figure = plt.figure(figsize=(24,24))
ax = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, alpha = 1.0)
ax.set_title('GOES EAST T6.7 (WV)', fontsize = 22)

# color bar

color = ax.pcolor(lon, lat, T67, vmin = -80, vmax = 40, cmap = 'jet')

cbar = figure.colorbar(color, orientation = 'horizontal', fraction = 0.05, pad = 0.02)
cbar.set_label("Temp(°C)", fontsize = 15)
cbar.ax.tick_params(labelsize = 12)

# plot channel 11 (IR)

figure = plt.figure(figsize=(24,24))
ax = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, alpha = 1.0)
ax.set_title('GOES EAST T11 (IR)', fontsize = 22)

# color bar

color = ax.pcolor(lon, lat, T11, vmin = -80, vmax = 40, cmap = 'jet')

cbar = figure.colorbar(color, orientation = 'horizontal', fraction = 0.05, pad = 0.02)
cbar.set_label("Temp(°C)", fontsize = 15)
cbar.ax.tick_params(labelsize = 12)

# plot difference

figure = plt.figure(figsize=(24,24))
ax = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, alpha = 1.0)
ax.set_title('GOES EAST T6.7 - T11 (Difference)', fontsize = 22)

# color bar

color = ax.pcolor(lon, lat, T67-T11, vmin = 0, vmax = 6, cmap = 'coolwarm')

cbar = figure.colorbar(color, orientation = 'horizontal', fraction = 0.05, pad = 0.02)
cbar.set_label("Temp(°C)", fontsize = 15)
cbar.ax.tick_params(labelsize = 12)



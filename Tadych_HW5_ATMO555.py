# %%# Importing relevant packages
import requests
import netCDF4
import numpy as np
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset 
import cartopy 
import cartopy.feature as cfeature
import cartopy.crs as ccrs

filename = "CER_GEO_Ed4_GOE16_NH_V01.2_2018.283.1230.06K.nc"

data = xr.open_dataset('Data/'+filename)
data

#%%
value1 = data.temperature_ir
#%%
metadata = data.attrs
metadata
# %% Slicing the data for T11 and T6.7 (meaning temp infrared and temp 6.7)
lat = data.variables['latitude'][:]
lon = data.variables['longitude'][:]
# time = data.variables['base_time'][:]
t11 = data['temperature_ir']
t67 = data['temperature_67']
t11
# %%
t67
# %%
difference = t11 - t67
# %%
t11[200:500,400:800].plot()
plt.savefig("Tadych_hw5_HurricaneMichael_t11")
# %%
t67[200:500,400:800].plot()
plt.savefig("Tadych_hw5_HurricaneMichael_t67")
#%%
difference[200:500,400:800].plot()
plt.savefig("Tadych_hw5_HurricaneMichael_diff")

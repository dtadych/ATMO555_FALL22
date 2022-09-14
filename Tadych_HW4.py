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

# %% --- Downloading the data from a text file ---

# opening the file in read mode
my_file = open("Data/subset_GPCPMON_3.2_20220913_231701.txt", "r")
  
# reading the file
data = my_file.read()
# %%  
# replacing end of line('/n') with ' ' and
# splitting the text it further when '.' is seen.
URLlist = data.split("\n")
  
# printing the data
print(URLlist)
my_file.close()

# %% gotta make a list of names
names = ["jan1984.nc4"
        ,"feb1984.nc4"
        ,"march.nc4"
        ,"apr1984.nc4"
        ,"may1984.nc4"
        ,"jun1984.nc4"
        ,"jul1984.nc4"
        ,"aug1984.nc4"
        ,"sep1984.nc4"
        ,"oct1984.nc4"
        ,"nov1984.nc4"
        ,"dec1984.nc4"]
# %%
for i,j in zip(URLlist,names):
    r = requests.get(i, stream = True)
    file = open(j,"wb")
    file.write(r.content)
    file.close()

# %% --- Reading in the data ---
for i in names:
    f = xr.open_dataset(i)
    print(f)

# %%
data = xr.open_dataset("Datajan1984.nc4")
data
# %%
file='Data/GPCPMON_L3_198401_V3.2.nc4'
data = xr.open_dataset(file)
data
# %%
names = [
        # 'GPCPMON_L3_198401_V3.2.nc4'
        'GPCPMON_L3_198406_V3.2.nc4'
        ,'GPCPMON_L3_198411_V3.2.nc4'
        ,'GPCPMON_L3_198402_V3.2.nc4'
        ,'GPCPMON_L3_198407_V3.2.nc4'
        ,'GPCPMON_L3_198412_V3.2.nc4'
        ,'GPCPMON_L3_198403_V3.2.nc4'              
        ,'GPCPMON_L3_198408_V3.2.nc4'
        ,'GPCPMON_L3_198404_V3.2.nc4'
        ,'GPCPMON_L3_198409_V3.2.nc4'
        ,'GPCPMON_L3_198405_V3.2.nc4'
        ,'GPCPMON_L3_198410_V3.2.nc4']

# %%
data = xr.open_dataset('Data/GPCPMON_L3_198401_V3.2.nc4')
data
# %%
for i in names:
    f = xr.open_dataset('Data/'+i)
    data = xr.merge([data,f])

# print(data)
# %% Slicing the dat
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
time = data.variables['time'][:]
precip = data['sat_gauge_precip']
print(precip)
# %%
global_mean = precip.mean(("lon","lat"))
global_mean.plot()
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

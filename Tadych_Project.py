   # Set the URL string to point to a specific data URL. Some generic examples are:
   #   https://servername/data/path/file
   #   https://servername/opendap/path/file[.format[?subset]]
   #   https://servername/daac-bin/OTF/HTTP_services.cgi?KEYWORD=value[&KEYWORD=value]
#%%
URL = 'https://n5eil02u.ecs.nsidc.org/esir/5000003786876.html'
   
# Set the FILENAME string to the data file name, the LABEL keyword value, or any customized name. 
FILENAME = '5000003786876.txt'
   
from glob import glob
from operator import index
import requests
result = requests.get(URL)
try:
   result.raise_for_status()
   f = open(FILENAME,'wb')
   f.write(result.content)
   f.close()
   print('contents of URL written to '+FILENAME)
except:
   print('requests.get() returned an error code '+str(result.status_code))
#%%
# %% Importing relevant packages
from re import S
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset 
import cartopy 
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import rasterio
import rioxarray as rxr
import pandas as pd
# import requests
import netCDF4

import scipy.stats as sp
from scipy.stats import kendalltau, pearsonr, spearmanr
import pymannkendall as mk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mlxtend.evaluate import bias_variance_decomp
import math
import h5py
import os
import datetime
# Some functions for analysis
def kendall_pval(x,y):
        return kendalltau(x,y)[1]
    
def pearsonr_pval(x,y):
        return pearsonr(x,y)[1]
    
def spearmanr_pval(x,y):
        return spearmanr(x,y)[1]

def display_correlation(df):
    r = df.corr(method="spearman")
    plt.figure(figsize=(10,6))
    heatmap = sns.heatmap(df.corr(method='spearman'), vmin=-1, 
                      vmax=1, annot=True)
    plt.title("Spearman Correlation")
    return(r)
#%%
datadirectory = 'Data/SMAP/'

A = ["SMAP_L3_SM_P_E_20220105_R18290_001_HEGOUT.nc"
    ,"SMAP_L3_SM_P_E_20220205_R18290_001_HEGOUT.nc"
    ,"SMAP_L3_SM_P_E_20220215_R18290_001_HEGOUT.nc"
    ,"SMAP_L3_SM_P_E_20220216_R18290_001_HEGOUT.nc"
    ,"SMAP_L3_SM_P_E_20220217_R18290_001_HEGOUT.nc"
    ,"SMAP_L3_SM_P_E_20220219_R18290_001_HEGOUT.nc"
    ,"SMAP_L3_SM_P_E_20220220_R18290_001_HEGOUT.nc"
    ,"SMAP_L3_SM_P_E_20220305_R18290_001_HEGOUT.nc"
    ,"SMAP_L3_SM_P_E_20220401_R18290_001_HEGOUT.nc"
    ,"SMAP_L3_SM_P_E_20220501_R18290_002_HEGOUT.nc"
    ,"SMAP_L3_SM_P_E_20220601_R18290_001_HEGOUT.nc"
    ,"SMAP_L3_SM_P_E_20220701_R18290_001_HEGOUT.nc"
    ,"SMAP_L3_SM_P_E_20220805_R18290_001_HEGOUT.nc"
    ,"SMAP_L3_SM_P_E_20220923_R18240_001_HEGOUT.nc"
    ,"SMAP_L3_SM_P_E_20221005_R18290_002_HEGOUT.nc"]

data_A = xr.open_dataset(datadirectory+str(A[0]))
data_A

for i in A:
    f = xr.open_dataset(datadirectory+i)
    data_A = xr.merge([data_A,f],compat='override')
# %%
data_A
# %%
data_A2 = xr.open_dataset(datadirectory+str(A[0]))
data_A2
# %%
datadirectory = 'Data/SMAP/'
fn = "SMAP_L3_SM_P_E_20221006_R18290_002_HEGOUT.nc"
data_rain = xr.open_dataset(datadirectory+fn)
#%%
data_rain[0].plot()
# %%
hd5directory = 'Data/SMAP/hd5/'
hd5_rain = [
            "SMAP_L3_SM_P_E_20220930_R18290_002.h5"
            ,"SMAP_L3_SM_P_E_20221001_R18290_002.h5"
            ,"SMAP_L3_SM_P_E_20221002_R18290_002.h5"
            ,"SMAP_L3_SM_P_E_20221003_R18290_001.h5"
            ,"SMAP_L3_SM_P_E_20221004_R18290_001.h5"
            ,"SMAP_L3_SM_P_E_20221005_R18290_002.h5"
            ,"SMAP_L3_SM_P_E_20221006_R18290_002.h5"
            ,"SMAP_L3_SM_P_E_20221007_R18290_001.h5"
            ,"SMAP_L3_SM_P_E_20221008_R18290_002.h5"
            ,"SMAP_L3_SM_P_E_20221009_R18290_001.h5"
            ,"SMAP_L3_SM_P_E_20221010_R18290_001.h5"
            ,"SMAP_L3_SM_P_E_20221011_R18290_001.h5"]

# %% Writing date time array
start = datetime.datetime(2022, 9, 30)
dt_array = np.array([start + datetime.timedelta(days=i) for i in range(12)])
print(dt_array)

# %%
file_name = hd5directory+hd5_rain[0]
f = h5py.File(file_name, 'r')

list(f.keys())
# %%
fs = f['Soil_Moisture_Retrieval_Data_AM']
list(fs.keys())
# %%
soilmoisture = fs['soil_moisture']
soilmoisture
# %%
soilmoisture_std = fs['soil_moisture_error']
soilmoisture_std

#%%
lat_main = fs['latitude']
lon_main = fs['longitude']
#%%
type(soilmoisture)
#%%
file_name = hd5directory+hd5_rain[0]
soilmoisture = f['Soil_Moisture_Retrieval_Data_AM/soil_moisture']
smdata = np.ndarray(shape=soilmoisture.shape,dtype=float)
soilmoisture.read_direct(smdata)
np.place(smdata, smdata==soilmoisture.attrs.get('_FillValue'), np.nan)
smdata
# %%
file_name = hd5directory+hd5_rain[5]
soilmoisture2 = f['Soil_Moisture_Retrieval_Data_AM/soil_moisture']
smdata2 = np.ndarray(shape=soilmoisture2.shape,dtype=float)
soilmoisture2.read_direct(smdata2)
np.place(smdata2, smdata2==soilmoisture2.attrs.get('_FillValue'), np.nan)

# %%
print(smdata2.shape)

# %% convert array to dataframe
smdf = pd.DataFrame(smdata)
smdf

# %%
alldata=[]
for i in range(12):
    file_name = hd5directory+hd5_rain[i]
    f = h5py.File(file_name, 'r')
    fs = f['Soil_Moisture_Retrieval_Data_AM']
    soilmoisture_wut = fs['soil_moisture']
    smdata_wut = np.ndarray(shape=soilmoisture_wut.shape,dtype=float)
    soilmoisture_wut.read_direct(smdata_wut)
    np.place(smdata_wut, smdata_wut==soilmoisture_wut.attrs.get('_FillValue'), np.nan)
    # combo = np.dstack((smdata,smdata_wut))
    alldata.append(smdata_wut)

alldata

# %%
combo = np.dstack(alldata)

combo.shape

# %% Create an xrray dataset
data_xr = xr.DataArray(
    combo,
    coords={'lat': lat_main[:,0],'lon': lon_main[0,:],'time':dt_array}, 
    dims=["lat", "lon","time"],
    attrs=dict(
        description="soil moisture",
        units=soilmoisture.attrs.get('units').decode('utf-8'))
)
data_xr
#%%
xrdataset = xr.DataArray.to_dataset(data_xr,name='Soil Moisture')
xrdataset

# %% Slicing the data again but idk somehow it's different now
lat = xrdataset.variables['lat'][:]
lon = xrdataset.variables['lon'][:]
time = xrdataset.variables['time'][:]
sm = xrdataset['Soil Moisture']

# %% Replace -9999 with NaN
# ds_masked = ds.where(ds['var'] != -9999.)  

#%% Assign coordinates
lwe = sm
lwe.coords['lon'] = (lwe.coords['lon']+180) % 360 - 180
lwe
# %%
lwe2 = lwe.sortby(lwe.lon)
lwe2 = lwe2.sortby(lwe2.lat)
#print(lwe2['lon'])
# %%
lwe2 = lwe2.rio.set_spatial_dims('lon', 'lat')
lwe2.rio.crs
# %%
lwe2 = lwe2.rio.set_crs("epsg:4269")
lwe2.rio.crs
# %%
lwe2
#%%
lwe2 = lwe2.sortby(lwe2.lat)
# %%
lwe2.plot()

# %%
global_mean_ts = lwe2.mean(("lon","lat"))
global_mean_ts.plot()

#%%
global_mean_rain = lwe2.mean(("time"))
# global_mean = global_mean.sortby(global_mean.lat)

# %%
Name = "Average Soil moisture over CONUS"
fig = plt.figure(figsize=(10,15))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
plt.title(Name)
global_mean_rain.plot(ax = ax)
# ax.set_extent([-114,-109,33,37]) # Just AZ
ax.set_extent([-130,-70,20,48]) # Narrowing on CONUS
# plt.savefig(Name, bbox_inches = 'tight')
# %% Slightly better plot from Earlier
data = global_mean_rain

Name = "Average Soil moisture over CONUS"
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
data.plot(ax = ax)
# ax.set_extent([-114,-109,33,37]) # Just Az
ax.set_extent([-125,-65,25,45]) # CONUS
# ax.set_extent([-116,-108,31,37.5]) # AZ with a lil buffer
ax.coastlines(resolution="50m",linewidth=0.5)
ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'))
ax.add_feature(cartopy.feature.STATES.with_scale('50m'), edgecolor='gray', facecolor='none')
gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.8,color='#555555',alpha=0.5,linestyle='--')

# Trying to control the colorbar
color = ax.pcolormesh(data.lon,data.lat,data,cmap='viridis')
cbar = fig.colorbar(color, ax=ax, orientation = 'horizontal')
# ax_color=ax.imshow(np.flip(data.transpose(), axis = 0),cmap='viridis')
# cbar = plt.colorbar(ax_color)
cbar.set_label("Soil Moisture (cm^3/cm^3)")

ax.set_title(Name, fontsize = 14)

# %% Now gotta get standard deviation

global_mean_rain_std = lwe2.std(("time"))
# global_mean = global_mean.sortby(global_mean.lat)

# %%
Name = "Average Soil moisture over CONUS"
fig = plt.figure(figsize=(10,15))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
plt.title(Name)
global_mean_rain_std.plot(ax = ax)
# ax.set_extent([-114,-109,33,37]) # Just AZ
ax.set_extent([-130,-70,20,48]) # Narrowing on CONUS
# plt.savefig(Name, bbox_inches = 'tight')
# %% Slightly better plot from Earlier
data = global_mean_rain_std

Name = "Standard Deviation of Soil moisture over CONUS"
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
data.plot(ax = ax)
# ax.set_extent([-114,-109,33,37]) # Just Az
ax.set_extent([-125,-65,25,45]) # CONUS
# ax.set_extent([-116,-108,31,37.5]) # AZ with a lil buffer
ax.coastlines(resolution="50m",linewidth=0.5)
ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'))
ax.add_feature(cartopy.feature.STATES.with_scale('50m'), edgecolor='gray', facecolor='none')
gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.8,color='#555555',alpha=0.5,linestyle='--')

# Trying to control the colorbar
color = ax.pcolormesh(data.lon,data.lat,data,cmap='RdPu')
cbar = fig.colorbar(color, ax=ax, orientation = 'horizontal')
# ax_color=ax.imshow(np.flip(data.transpose(), axis = 0),cmap='viridis')
# cbar = plt.colorbar(ax_color)
# cbar.set_label("Soil Moisture (cm^3/cm^3)")

ax.set_title(Name, fontsize = 14)

# %% Getting timeseries just for AZ

global_mean_rain_2 = global_mean_rain.sortby(global_mean_rain.lat)
# %% Bounding box goal = [-115,-109,31,37]
az_rain = global_mean_rain_2[1334:1406,625:694]
# global_mean_rain_2[1343:1415,694:759]

az_rain.plot()

# %%
lwe2[:,:,5:6].plot()

# %%
azrain_ts = lwe2[1334:1406,625:694].mean(("lon","lat"))
azrain_ts.plot()

#%% Plotting at Average between 2 and 4 km
# Subset the 2-Dimensional variable over the eye of Hurricane Harvey
# Choose the range of the subset, e.g.:
#   170 rows in the along-track dimension,
#   All 49 elements in the cross-track dimension,
start = 0
end = 3856
# mysub = prnsdata[start:end,:]
mysub = smdata2[:,:]
mylon = lon_main[:,:]
mylat = lat_main[:,:]

# Draw the subset of near-surface precipitation rate 
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-130,-60,20,50])
plt.title(hd5_rain[5])

# Add coastlines and gridlines
ax.coastlines(resolution="50m",linewidth=0.5)
gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.8,color='#555555',alpha=0.5,linestyle='--')

# Axis labels
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = True

# Plot the scatter diagram 
pp = plt.scatter(mylon, mylat, c=mysub, cmap=plt.cm.jet, transform=ccrs.PlateCarree())

# Add a colorbar to the bottom of the plot.
fig.subplots_adjust(bottom=0.15,left=0.06,right=0.94)
cbar_ax = fig.add_axes([0.12, 0.11, 0.76, 0.015])  
cbar = plt.colorbar(pp,cax=cbar_ax,orientation='horizontal')
cbar.set_label(label=soilmoisture.attrs.get('units').decode('utf-8'),size=10)
# %%
print(range(12))
print(len(hd5_rain))
# %%
for i in hd5_rain:
    file_name = hd5directory+i
    soilmoisture2 = f['Soil_Moisture_Retrieval_Data_AM/soil_moisture']
    smdata2 = np.ndarray(shape=soilmoisture2.shape,dtype=float)
    soilmoisture2.read_direct(smdata2)
    np.place(smdata2, smdata2==soilmoisture2.attrs.get('_FillValue'), np.nan)
    mysub = smdata2[:,:]
    mylon = lon[:,:]
    mylat = lat[:,:]

    # Draw the subset of near-surface precipitation rate 
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-130,-60,20,50])
    plt.title(file_name)

    # Add coastlines and gridlines
    ax.coastlines(resolution="50m",linewidth=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.8,color='#555555',alpha=0.5,linestyle='--')

    # Axis labels
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = True

    # Plot the scatter diagram 
    pp = plt.scatter(mylon, mylat, c=mysub, cmap=plt.cm.jet, transform=ccrs.PlateCarree())

    # Add a colorbar to the bottom of the plot.
    fig.subplots_adjust(bottom=0.15,left=0.06,right=0.94)
    cbar_ax = fig.add_axes([0.12, 0.11, 0.76, 0.015])  
    cbar = plt.colorbar(pp,cax=cbar_ax,orientation='horizontal')
    cbar.set_label(label=soilmoisture.attrs.get('units').decode('utf-8'),size=10)
    plt.savefig(i, bbox_inches = 'tight')

# %% Need to get a timeseries plot for PM data
pmsoilmoisture = f['Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm']
pmsoilmoisture
# %%
fp = f['Soil_Moisture_Retrieval_Data_PM']
list(fp.keys())
# %%
pmsmdata = np.ndarray(shape=pmsoilmoisture.shape,dtype=float)
pmsoilmoisture.read_direct(pmsmdata)
np.place(pmsmdata, pmsmdata==pmsoilmoisture.attrs.get('_FillValue'), np.nan)

# %%
#%% This is just a generic but pretty array map plot
start = 0
end = 3856
# mysub = prnsdata[start:end,:]
mysub = pmsmdata[start:end,:]
mylon = lon_main[start:end,:]
mylat = lat_main[start:end,:]

# Draw the subset of near-surface precipitation rate 
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-130,-70,20,50])
# plt.title('Average Precipitation Rate at 2km from GPM_2ADPR, 25 August 2017')

# Add coastlines and gridlines
ax.coastlines(resolution="50m",linewidth=0.5)
gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.8,color='#555555',alpha=0.5,linestyle='--')

# Axis labels
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = True

# Plot the scatter diagram 
pp = plt.scatter(mylon, mylat, c=mysub, cmap=plt.cm.jet, transform=ccrs.PlateCarree())

# Add a colorbar to the bottom of the plot.
fig.subplots_adjust(bottom=0.15,left=0.06,right=0.94)
cbar_ax = fig.add_axes([0.12, 0.11, 0.76, 0.015])  
cbar = plt.colorbar(pp,cax=cbar_ax,orientation='horizontal')
cbar.set_label(label=pmsoilmoisture.attrs.get('units').decode('utf-8'),size=10)

# %% Now to do the PM data

alldata=[]
for i in range(12):
    file_name = hd5directory+hd5_rain[i]
    f = h5py.File(file_name, 'r')
    soilmoisture_wut = f['Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm']
    smdata_wut = np.ndarray(shape=soilmoisture_wut.shape,dtype=float)
    soilmoisture_wut.read_direct(smdata_wut)
    np.place(smdata_wut, smdata_wut==soilmoisture_wut.attrs.get('_FillValue'), np.nan)
    # combo = np.dstack((smdata,smdata_wut))
    alldata.append(smdata_wut)

combo = np.dstack(alldata)

# Create an xrray dataset
data_xr = xr.DataArray(
    combo,
    coords={'lat': lat_main[:,0],'lon': lon_main[0,:],'time':dt_array}, 
    dims=["lat", "lon","time"],
    attrs=dict(
        description="soil moisture",
        units=soilmoisture.attrs.get('units').decode('utf-8'))
)

xrdataset = xr.DataArray.to_dataset(data_xr,name='Soil Moisture')
xrdataset

# %% Slicing the data again but idk somehow it's different now
lat = xrdataset.variables['lat'][:]
lon = xrdataset.variables['lon'][:]
time = xrdataset.variables['time'][:]
sm = xrdataset['Soil Moisture']

# Assign coordinates
lwe = sm
lwe.coords['lon'] = (lwe.coords['lon']+180) % 360 - 180

lwe2 = lwe.sortby(lwe.lon)
lwe2 = lwe2.sortby(lwe2.lat)
#print(lwe2['lon'])

lwe2 = lwe2.rio.set_spatial_dims('lon', 'lat')

lwe2 = lwe2.rio.set_crs("epsg:4269")
lwe2.rio.crs
#%%
lwe2 = lwe2.sortby(lwe2.lat)
lwe2.plot()

#%%
global_mean_rain_pm = lwe2.mean(("time"))
# global_mean = global_mean.sortby(global_mean.lat)

# %% Getting timeseries just for AZ

global_mean_rain_2 = global_mean_rain.sortby(global_mean_rain.lat)
# %% Bounding box goal = [-115,-109,31,37]
az_rain_pm = global_mean_rain_2[1334:1406,625:694]
# global_mean_rain_2[1343:1415,694:759]

az_rain_pm.plot()

# %%
lwe2[:,:,5:6].plot()

# %%
azrain_ts_pm = lwe2[1334:1406,625:694].mean(("lon","lat"))
azrain_ts_pm.plot()

# %%
import matplotlib.dates as mdates
# %% Converting to dataframes
azrain_amdf = pd.DataFrame(azrain_ts)
azrain_amdf['Date'] = dt_array
azrain_amdf = azrain_amdf.set_index('Date')
azrain_amdf

# %%
azrain_pmdf = pd.DataFrame(azrain_ts_pm)
azrain_pmdf['Date'] = dt_array
azrain_pmdf = azrain_pmdf.set_index('Date')
azrain_pmdf
# %% Plotting AM Versus PM
f, ax = plt.subplots(figsize=(8, 5))

ax.plot(azrain_amdf, color='red', label='AM')
ax.plot(azrain_pmdf, color='black', label = 'PM')

ax.grid(visible=True,which='major')
name = "Average AM versus PM Soil Moisture Values"
ax.set(title=name)
ax.legend(loc='lower left')
plt.xlabel('Date')
plt.ylabel('Soil Moisture (cm^3/cm^3)')
fig.set_dpi(600.0)
plt.savefig(name)

# %%
average_ampm = azrain_amdf.add(azrain_pmdf)
average_ampm = average_ampm / 2
average_ampm.iloc[1] = azrain_amdf.iloc[1]
average_ampm.iloc[5] = azrain_pmdf.iloc[5]
average_ampm.iloc[9] = azrain_amdf.iloc[9]
average_ampm

# %% Plotting AM Versus PM
f, ax = plt.subplots(figsize=(8, 5))

ax.plot(azrain_amdf, color='red', label='AM')
ax.plot(azrain_pmdf, color='black', label = 'PM')
ax.plot(average_ampm,'--',color='grey', label = 'Average')

ax.grid(visible=True,which='major')
name = "Average AM versus PM Soil Moisture Values"
ax.set(title=name)
ax.legend(loc='lower left')
plt.xlabel('Date')
plt.ylabel('Soil Moisture (cm^3/cm^3)')
fig.set_dpi(600.0)
plt.savefig(name)

# %% 5. Now to make a map of the flagsies
alldata=[]
for i in range(12):
    file_name = hd5directory+hd5_rain[i]
    f = h5py.File(file_name, 'r')
    soilmoisture_wut = f['Soil_Moisture_Retrieval_Data_AM/retrieval_qual_flag']
    smdata_wut = np.ndarray(shape=soilmoisture_wut.shape,dtype=float)
    soilmoisture_wut.read_direct(smdata_wut)
    np.place(smdata_wut, smdata_wut==soilmoisture_wut.attrs.get('_FillValue'), np.nan)
    # combo = np.dstack((smdata,smdata_wut))
    alldata.append(smdata_wut)

combo = np.dstack(alldata)

# Create an xrray dataset
data_xr = xr.DataArray(
    combo,
    coords={'lat': lat_main[:,0],'lon': lon_main[0,:],'time':dt_array}, 
    dims=["lat", "lon","time"],
    attrs=dict(
        description="Bit flags that record the conditions and the quality of the retrieval algorithsm that generate soil moisture for the grid cell",
        units='quality (0=Good, 1=Bad')
)

xrdataset = xr.DataArray.to_dataset(data_xr,name='Soil Moisture Flag')
xrdataset

# %% Slicing the data again but idk somehow it's different now
lat = xrdataset.variables['lat'][:]
lon = xrdataset.variables['lon'][:]
time = xrdataset.variables['time'][:]
sm = xrdataset['Soil Moisture Flag']

# Assign coordinates
lwe = sm
lwe.coords['lon'] = (lwe.coords['lon']+180) % 360 - 180

lwe2 = lwe.sortby(lwe.lon)
lwe2 = lwe2.sortby(lwe2.lat)
#print(lwe2['lon'])

lwe2 = lwe2.rio.set_spatial_dims('lon', 'lat')

lwe2 = lwe2.rio.set_crs("epsg:4269")
lwe2.rio.crs
#%%
lwe2 = lwe2.sortby(lwe2.lat)
lwe2.plot()

#%%

# ds_masked = ds.where(ds['var'] != -9999.)  

# global_mean_quality = lwe2.mean(("time"))
# global_mean = global_mean.sortby(global_mean.lat)
global_mean_quality = lwe2.where(lwe2 == 0)
global_mean_quality = global_mean_quality.mean(('time'))
global_mean_quality

# %%
Name = "Average Soil moisture over CONUS"
fig = plt.figure(figsize=(10,15))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
plt.title(Name)
global_mean_quality.plot(ax = ax)
# ax.set_extent([-114,-109,33,37]) # Just AZ
ax.set_extent([-130,-70,20,48]) # Narrowing on CONUS
# plt.savefig(Name, bbox_inches = 'tight')
# %% Slightly better plot from Earlier
data = global_mean_quality

Name = "Soil Moisture Quality during Time Period (9/30 - 10/11, 2022) \n \n0 (Green) = Good"
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
data.plot(ax = ax)
# ax.set_extent([-114,-109,33,37]) # Just Az
# ax.set_extent([-125,-65,25,45]) # CONUS
ax.set_extent([-115,-109,31,37.5]) # AZ with a lil buffer
ax.coastlines(resolution="50m",linewidth=0.5)
ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'))
ax.add_feature(cartopy.feature.STATES.with_scale('50m'), edgecolor='blue', facecolor='none')
gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.8,color='#555555',alpha=0.5,linestyle='--')

# Trying to control the colorbar
color = ax.pcolormesh(data.lon,data.lat,data,cmap='gist_earth')
cbar = fig.colorbar(color, ax=ax, orientation = 'horizontal')
# ax_color=ax.imshow(np.flip(data.transpose(), axis = 0),cmap='viridis')
# cbar = plt.colorbar(ax_color)
cbar.set_label("0 (Green) = Good")

ax.set_title(Name, fontsize = 14)

# %% ---- Do this for 2022 data ----

# create list of files from directory
path = hd5directory+'2022/'
items = [f for f in os.listdir(path) if os.path.isfile( os.path.join(path, f) )]
items

# %%
hd5_2022 = items
hd5_2022.sort()
hd5_2022

# %%
start = datetime.datetime(2022, 11, 1)
dt_array = np.array([start + datetime.timedelta(days=i) for i in range(len(hd5_2022))])
print(dt_array)

# %%
alldata=[]
for i in range(len(hd5_2022)):
    file_name = hd5directory+'2022/'+hd5_2022[i]
    f = h5py.File(file_name, 'r')
    fs = f['Soil_Moisture_Retrieval_Data_AM']
    soilmoisture_wut = fs['soil_moisture']
    smdata_wut = np.ndarray(shape=soilmoisture_wut.shape,dtype=float)
    soilmoisture_wut.read_direct(smdata_wut)
    np.place(smdata_wut, smdata_wut==soilmoisture_wut.attrs.get('_FillValue'), np.nan)
    # combo = np.dstack((smdata,smdata_wut))
    alldata.append(smdata_wut)

alldata

# %%
combo = np.dstack(alldata)

combo.shape

# %% Create an xrray dataset

lat = fs['latitude']
lon = fs['longitude']

data_xr = xr.DataArray(
    combo,
    coords={'lat': lat[:,0],'lon': lon[0,:],'time':dt_array}, 
    dims=["lat", "lon","time"],
    attrs=dict(
        description="soil moisture",
        units=soilmoisture.attrs.get('units').decode('utf-8'))
)
data_xr
#%%
xrdataset = xr.DataArray.to_dataset(data_xr,name='Soil Moisture')
xrdataset

# %% Slicing the data again but idk somehow it's different now
lat = xrdataset.variables['lat'][:]
lon = xrdataset.variables['lon'][:]
time = xrdataset.variables['time'][:]
sm = xrdataset['Soil Moisture']

# %% Replace -9999 with NaN
# ds_masked = ds.where(ds['var'] != -9999.)  

#%% Assign coordinates
lwe = sm
lwe.coords['lon'] = (lwe.coords['lon']+180) % 360 - 180
lwe
# %%
lwe2 = lwe.sortby(lwe.lon)
lwe2 = lwe2.sortby(lwe2.lat)
#print(lwe2['lon'])
# %%
lwe2 = lwe2.rio.set_spatial_dims('lon', 'lat')
lwe2.rio.crs
# %%
lwe2 = lwe2.rio.set_crs("epsg:4269")
lwe2.rio.crs
# %%
lwe2
#%%
lwe2 = lwe2.sortby(lwe2.lat)
# %%
lwe2.plot()

# %%
global_mean_ts_2022 = lwe2.mean(("lon","lat"))
global_mean_ts_2022.plot()

#%%
global_mean_2022 = lwe2.mean(("time"))
# global_mean = global_mean.sortby(global_mean.lat)

# %%
data = global_mean_2022

Name = 'Average Soil Moisture for 11/2022 \n(SMAP L3 9km)'
fig = plt.figure(figsize=(10,15))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
data.plot(ax = ax)
# ax.set_extent([-114,-109,33,37]) # Just Az
# ax.set_extent([-120,-105,20,40]) # CONUS
ax.set_extent([-116,-108,31,37.5]) # AZ with a lil buffer
ax.coastlines(resolution="50m",linewidth=0.5)
ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'))
ax.add_feature(cartopy.feature.STATES.with_scale('50m'), edgecolor='gray', facecolor='none')
gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.8,color='#555555',alpha=0.5,linestyle='--')

# Trying to control the colorbar
color = ax.pcolormesh(data.lon,data.lat,data,cmap='viridis')
cbar = fig.colorbar(color, ax=ax, orientation = 'horizontal')
# ax_color=ax.imshow(np.flip(data.transpose(), axis = 0),cmap='viridis')
# cbar = plt.colorbar(ax_color)
cbar.set_label("Soil Moisture (cm^3/cm^3)")

ax.set_title(Name, fontsize = 14)
plt.savefig(hd5directory+Name, bbox_inches = 'tight')

# %% ---- Do this for 2015 data ----
# create list of files from directory
path = hd5directory+'2015/'
items = [f for f in os.listdir(path) if os.path.isfile( os.path.join(path, f) )]
items

# %% Make datetime array
start = datetime.datetime(2015, 9, 21)
dt_array = np.array([start + datetime.timedelta(days=i) for i in range(len(items))])
print(dt_array)

# %%
items.sort()
items

# %%
alldata=[]
for i in range(len(items)):
    file_name = path+items[i]
    f = h5py.File(file_name, 'r')
    fs = f['Soil_Moisture_Retrieval_Data_AM']
    soilmoisture_wut = fs['soil_moisture']
    smdata_wut = np.ndarray(shape=soilmoisture_wut.shape,dtype=float)
    soilmoisture_wut.read_direct(smdata_wut)
    np.place(smdata_wut, smdata_wut==soilmoisture_wut.attrs.get('_FillValue'), np.nan)
    # combo = np.dstack((smdata,smdata_wut))
    alldata.append(smdata_wut)

alldata

# %%
combo = np.dstack(alldata)

combo.shape

# %% Create an xrray dataset

lat = fs['latitude']
lon = fs['longitude']

data_xr = xr.DataArray(
    combo,
    coords={'lat': lat[:,0],'lon': lon[0,:],'time':dt_array}, 
    dims=["lat", "lon","time"],
    attrs=dict(
        description="soil moisture",
        units=soilmoisture.attrs.get('units').decode('utf-8'))
)
data_xr
#%%
xrdataset = xr.DataArray.to_dataset(data_xr,name='Soil Moisture')
xrdataset

# %% Slicing the data again but idk somehow it's different now
lat = xrdataset.variables['lat'][:]
lon = xrdataset.variables['lon'][:]
time = xrdataset.variables['time'][:]
sm = xrdataset['Soil Moisture']

# %% Replace -9999 with NaN
# ds_masked = ds.where(ds['var'] != -9999.)  

#%% Assign coordinates
lwe = sm
lwe.coords['lon'] = (lwe.coords['lon']+180) % 360 - 180
lwe
# %%
lwe2 = lwe.sortby(lwe.lon)
lwe2 = lwe2.sortby(lwe2.lat)
#print(lwe2['lon'])
# %%
lwe2 = lwe2.rio.set_spatial_dims('lon', 'lat')
lwe2.rio.crs
# %%
lwe2 = lwe2.rio.set_crs("epsg:4269")
lwe2.rio.crs
# %%
lwe2
#%%
lwe2 = lwe2.sortby(lwe2.lat)
# %%
lwe2.plot()

# %%
global_mean_ts_2015 = lwe2.mean(("lon","lat"))
global_mean_ts_2015.plot()

#%%
global_mean_2015 = lwe2.mean(("time"))
# global_mean = global_mean.sortby(global_mean.lat)

# %% Simple weird plot but keeping for sciencing
fig = plt.figure(figsize=(10,15))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
global_mean_2015.plot(ax = ax)
# # ax.set_extent([-114,-109,33,37]) # Just AZ
ax.set_extent([-130,-70,20,50]) # CONUS

# %%
data = global_mean_2015

Name = 'Average Soil Moisture for 2015 \n(SMAP L3 9km)'
fig = plt.figure(figsize=(10,15))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
data.plot(ax = ax)
# ax.set_extent([-114,-109,33,37]) # Just Az
# ax.set_extent([-120,-105,20,40]) # CONUS
ax.set_extent([-116,-108,31,37.5]) # AZ with a lil buffer
ax.coastlines(resolution="50m",linewidth=0.5)
ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'))
ax.add_feature(cartopy.feature.STATES.with_scale('50m'), edgecolor='gray', facecolor='none')
gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.8,color='#555555',alpha=0.5,linestyle='--')

# Trying to control the colorbar
color = ax.pcolormesh(data.lon,data.lat,data,cmap='viridis')
cbar = fig.colorbar(color, ax=ax, orientation = 'horizontal')
# ax_color=ax.imshow(np.flip(data.transpose(), axis = 0),cmap='viridis')
# cbar = plt.colorbar(ax_color)
cbar.set_label("Soil Moisture (cm^3/cm^3)")

ax.set_title(Name, fontsize = 14)
# plt.savefig(hd5directory+Name, bbox_inches = 'tight')

# %% --- selecting out AZ and getting the difference
global_mean_2022_2 = global_mean_2022.sortby(global_mean_2022.lat)
# %% Bounding box goal = [-115,-109,31,37]
az_22 = global_mean_2022_2[1343:1415,694:759]
az_22.plot()
# %%
global_mean_2015_2 = global_mean_2015.sortby(global_mean_2015.lat)
# %%
az_15 = global_mean_2015_2[1230:1302,692:757]
az_15.plot()
# %% Getting the difference between 2015 and 2022
difference = az_22 - az_15
difference
# %%
difference.plot()

# %%
data = difference

Name = 'Difference Between 2015 and 2022'
fig = plt.figure(figsize=(10,15))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
data.plot(ax = ax)
# ax.set_extent([-114,-109,33,37]) # Just Az
# ax.set_extent([-120,-105,20,40]) # CONUS
ax.set_extent([-116,-108,31,37.5]) # AZ with a lil buffer
ax.coastlines(resolution="50m",linewidth=0.5)
ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'))
ax.add_feature(cartopy.feature.STATES.with_scale('50m'), edgecolor='gray', facecolor='none')
gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.8,color='#555555',alpha=0.5,linestyle='--')

# Trying to control the colorbar
color = ax.pcolormesh(data.lon,data.lat,data,cmap='RdBu')
cbar = fig.colorbar(color, ax=ax, orientation = 'horizontal')
# ax_color=ax.imshow(np.flip(data.transpose(), axis = 0),cmap='viridis')
# cbar = plt.colorbar(ax_color)
cbar.set_label("Soil Moisture (cm^3/cm^3)")

ax.set_title(Name, fontsize = 14)

# %% For plotting individual files
file_name = hd5directory+'2015/'+hd5_2015[0]
soilmoisture2 = f['Soil_Moisture_Retrieval_Data_AM/soil_moisture']
smdata2 = np.ndarray(shape=soilmoisture2.shape,dtype=float)
soilmoisture2.read_direct(smdata2)
np.place(smdata2, smdata2==soilmoisture2.attrs.get('_FillValue'), np.nan)

lat = fs['latitude']
lon = fs['longitude']

start = 0
end = 3856
# mysub = prnsdata[start:end,:]
mysub = smdata2[start:end,:]
mylon = lon[start:end,:]
mylat = lat[start:end,:]

# Draw the subset of near-surface precipitation rate 
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-130,-70,20,50])
# plt.title('Average Precipitation Rate at 2km from GPM_2ADPR, 25 August 2017')

# Add coastlines and gridlines
ax.coastlines(resolution="50m",linewidth=0.5)
gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.8,color='#555555',alpha=0.5,linestyle='--')

# Axis labels
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = True

# Plot the scatter diagram 
pp = plt.scatter(mylon, mylat, c=mysub, cmap=plt.cm.jet, transform=ccrs.PlateCarree())

# Add a colorbar to the bottom of the plot.
fig.subplots_adjust(bottom=0.15,left=0.06,right=0.94)
cbar_ax = fig.add_axes([0.12, 0.11, 0.76, 0.015])  
cbar = plt.colorbar(pp,cax=cbar_ax,orientation='horizontal')
cbar.set_label(label=pmsoilmoisture.attrs.get('units').decode('utf-8'),size=10)

# %%
path = hd5directory+'2015/'
items = [f for f in os.listdir(path) if os.path.isfile( os.path.join(path, f) )]
items
# %%

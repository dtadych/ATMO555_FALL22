# Homework 9
# Danielle Tadych
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
import gpm

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


# %%
datadirectory = 'Data/GMIDPR/'
hd5directory = 'Data/GMIDPR/hd5/'

# %% gotta make a list of names
A = ["2A.GPM.DPR.V9-20211125.20170825-S111924-E125156.019833.V07A.HDF5.nc4"
        ,"2A.GPM.DPR.V9-20211125.20170825-S203442-E220714.019839.V07A.HDF5.nc4"]
B = ["2B.GPM.DPRGMI.CORRA2022.20170825-S111924-E125156.019833.V07A.HDF5.nc4"
        ,"2B.GPM.DPRGMI.CORRA2022.20170825-S203442-E220714.019839.V07A.HDF5.nc4"]

hd5_A = ["2A.GPM.DPR.V9-20211125.20170825-S111924-E125156.019833.V07A.HDF5"
        ,"2A.GPM.DPR.V9-20211125.20170825-S203442-E220714.019839.V07A.HDF5"]
hd5_B = ["2B.GPM.DPRGMI.CORRA2022.20170825-S111924-E125156.019833.V07A.HDF5"
        ,"2B.GPM.DPRGMI.CORRA2022.20170825-S203442-E220714.019839.V07A.HDF5"]


# %% --- Reading in the data ---
data_A = xr.open_dataset(datadirectory+str(A[0]))
data_A

data_B = xr.open_dataset(datadirectory+str(B[0]))
data_B


# %% --- Reading in the data ---
# List Contents
gpm.listContents(hd5directory+hd5_A[0])

#%%
gpm.readSwath(hd5directory+hd5_A[0],'precipRate','GPM')

# %% This makes them all come together
for i in A:
    f = xr.open_dataset(datadirectory+i)
    data_A = xr.merge([data_A,f],compat='override')

for i in B:
    f = xr.open_dataset(datadirectory+i)
    data_B = xr.merge([data_B,f], compat='override')


print(data_A)

# %% ----- Tutorial from Nasa -----
# https://hdfeos.org/zoo/GESDISC/2A.GPM.DPR.V7-20170308.20170704-S001905-E015140.019017.V05A.HDF5.py
# it's not working

import os

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
#%%
# Reduce font size because file name is long.
mpl.rcParams.update({'font.size': 8})


file_name = hd5directory+hd5_A[0]

with h5py.File(file_name, mode='r') as f:
    
    name = '/FS/SLV/precipRate'
    data = f[name][:]
    units = f[name].attrs['units']
    _FillValue = f[name].attrs['_FillValue']
    data[data == _FillValue] = np.nan
    data = np.ma.masked_where(np.isnan(data), data)
        
    # Get the geolocation data.
    latitude = f['/FS/Latitude'][:]
    longitude = f['/FS/Longitude'][:]

        
    m = Basemap(projection='cyl', resolution='l',
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180)
    m.drawcoastlines(linewidth=0.5)
    m.drawparallels(np.arange(-90, 91, 45))
    m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])
    m.scatter(longitude, latitude, c=data, s=1, cmap=plt.cm.jet,
              edgecolors=None, linewidth=0)
    cb = m.colorbar(location="bottom", pad='10%')    
    cb.set_label(units)

    basename = os.path.basename(file_name)
    plt.title('{0}\n{1}'.format(basename, name))
    fig = plt.gcf()
    # plt.show()
    pngfile = "{0}.py.png".format(basename)
#     fig.savefig(pngfile)

# %%
file_name = hd5directory+hd5_A[0]
f = h5py.File(file_name, 'r')
f

# %%
list(f.keys())

# %%
fs = f['FS']
fs
#%%
list(fs.keys())

#%%
SLV = fs['SLV']
SLV.keys()
#%%
preciprate = SLV['precipRate'][()] # Saves it as an array


#%%
type(preciprate)
#%%
plt.imshow(preciprate[30])
plt.show()

# ----
#%%
fn = '2A.GPM.DPR.V9-20211125.20170825-S111924-E125156.019833.V07A (1).HDF5'
f = h5py.File(hd5directory+fn,'r')
f
#%%
fb = h5py.File(hd5directory+hd5_B[0],'r')
fb
#%%
fb['KuKaGMI'].keys()

#%%
lat = f['/FS/Latitude']
lon = f['/FS/Longitude']
preciprate = f['FS/SLV/precipRate']
nearsurf = f['FS/SLV/precipRateNearSurface']
height = f['FS/PRE/heightStormTop']
precip2km = f['FS/SLV/precipRateAve24']
cross = f['FS/SLV/zFactorFinal'][:,:,:,1]
#%%
gmi = fb['KuKaGMI/precipTotRate']
latb = fb['KuKaGMI/Latitude']
lonb = fb['KuKaGMI/Longitude']

print(gmi.shape)
#%%
file_name = hd5directory+hd5_A[0]
f = h5py.File(file_name, 'r')

# %%
print('/FS/SLV/precipRate dimension sizes are:',preciprate.shape)
#%%
print('Cross track shape is ', cross.shape)
# %%
# Read the data into numpy arrays and put NaN where FillValues are
# 3-Dimensional Precipitation Rate
prdata = np.ndarray(shape=preciprate.shape,dtype=float)
preciprate.read_direct(prdata)
np.place(prdata, prdata==preciprate.attrs.get('_FillValue'), np.nan)
# %%
# 2-Dimensional Near-Surface Precipitation Rage
prnsdata = np.ndarray(shape=nearsurf.shape,dtype=float)
nearsurf.read_direct(prnsdata)
np.place(prnsdata, prnsdata==nearsurf.attrs.get('_FillValue'), np.nan)

# %%
# 2-Dimensional Near-Surface Precipitation Rage
twokmdata = np.ndarray(shape=precip2km.shape,dtype=float)
precip2km.read_direct(twokmdata)
np.place(twokmdata, twokmdata==precip2km.attrs.get('_FillValue'), np.nan)

# %%
# 2-Dimensional Near-Surface Precipitation Rage
crosstrack = np.ndarray(shape=cross.shape,dtype=float)
#%%
# cross.read_direct(crosstrack)
np.place(crosstrack, crosstrack==cross.attrs.get('_FillValue'), np.nan)
print('Cross track shape is ', crosstrack.shape)

# %%
# 2-Dimensional Near-Surface Precipitation Rage
stormheight = np.ndarray(shape=height.shape,dtype=float)
height.read_direct(stormheight)
np.place(stormheight, stormheight==height.attrs.get('_FillValue'), np.nan)

#%% Plotting near surface
# Subset the 2-Dimensional variable over the eye of Hurricane Harvey
# Choose the range of the subset, e.g.:
#   170 rows in the along-track dimension,
#   All 49 elements in the cross-track dimension,
start = 2500
end = 2700
# mysub = prnsdata[start:end,:]
mysub = gmidata[start:end,:]
mylon = lonb[start:end,:]
mylat = latb[start:end,:]

# Draw the subset of near-surface precipitation rate 
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-100,-85,20,35])
plt.title('Near-Surface Precipitation Rate from GPM_2ADPR, 25 August 2017')

# Add coastlines and gridlines
ax.coastlines(resolution="50m",linewidth=0.5)
gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.8,color='#555555',alpha=0.5,linestyle='--')

# Axis labels
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = True

# Plot the scatter diagram 
pp = plt.scatter(mylon, mylat, c=mysub, cmap=plt.cm.viridis, transform=ccrs.PlateCarree())

# Add a colorbar to the bottom of the plot.
fig.subplots_adjust(bottom=0.15,left=0.06,right=0.94)
cbar_ax = fig.add_axes([0.12, 0.11, 0.76, 0.025])  
cbar = plt.colorbar(pp,cax=cbar_ax,orientation='horizontal')
cbar.set_label(label=gmi.attrs.get('units').decode('utf-8'),size=10)

#%%
print(mysub.max())
#%% Plotting at Average between 2 and 4 km
# Subset the 2-Dimensional variable over the eye of Hurricane Harvey
# Choose the range of the subset, e.g.:
#   170 rows in the along-track dimension,
#   All 49 elements in the cross-track dimension,
start = 2500
end = 2700
# mysub = prnsdata[start:end,:]
mysub = prdata[start:end,:]
mylon = lon[start:end,:]
mylat = lat[start:end,:]

# Draw the subset of near-surface precipitation rate 
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-100,-85,20,35])
plt.title('Average Precipitation Rate at 2km from GPM_2ADPR, 25 August 2017')

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
cbar.set_label(label=precip2km.attrs.get('units').decode('utf-8'),size=10)

#%% Calculating statistics for part 3
difference = prnsdata - twokmdata
bias = prnsdata.mean() - twokmdata.mean()

print('bias is ', int(bias))

print('Pearson Correlation coefficient')
df1 = prnsdata
df2 = twokmdata
x=pd.DataFrame(df1)
y=pd.DataFrame(df2)
r = sp.pearsonr(x.all(),y.all())
print(r)

#%% Plotting the difference (Part 3)
# Subset the 2-Dimensional variable over the eye of Hurricane Harvey
# Choose the range of the subset, e.g.:
#   170 rows in the along-track dimension,
#   All 49 elements in the cross-track dimension,
start = 2500
end = 2700
# mysub = prnsdata[start:end,:]
mysub = difference[start:end,:]
mylon = lon[start:end,:]
mylat = lat[start:end,:]

# Draw the subset of near-surface precipitation rate 
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-100,-85,20,35])
plt.title('Difference between near surface and 2km above from GPM_2ADPR, 25 August 2017')

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
cbar.set_label(label=precip2km.attrs.get('units').decode('utf-8'),size=10)

#%% Plotting height at storm top (Part 4)
# Subset the 2-Dimensional variable over the eye of Hurricane Harvey
# Choose the range of the subset, e.g.:
#   170 rows in the along-track dimension,
#   All 49 elements in the cross-track dimension,
start = 2500
end = 2680
# mysub = prnsdata[start:end,:]
mysub = stormheight[start:end,:]
mylon = lon[start:end,:]
mylat = lat[start:end,:]

# Draw the subset of near-surface precipitation rate 
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-100,-85,20,35])
plt.title('Height at Storm Top from GPM_2ADPR, 25 August 2017')

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
cbar.set_label(label=height.attrs.get('units').decode('utf-8'),size=10)

#%%
print(mysub.max())
# %%
# ALONG-TRACK VERTICAL PROFILE
# Choose the range of the subset, e.g.:
#   170 rows in the along-track dimension,
#   One element in the cross-track dimension that slices through the eye of the storm
#   The lowest 76 elements in the vertical dimension
start = 2679
end = 2680
mysub2 = cross[start:end,:,:]
mysub2[mysub2==-9999.9]=0
# Transpose the array so the along-track dimension will be the X-Axis
#   and the vertical dimension will be the Y-Axis
subset2 = np.transpose(mysub2)

# Draw the ALONG-TRACK VERTICAL PROFILE
fig = plt.figure(figsize=(8,6))
ax = plt.axes()
ax.set_xlabel('along-track axis index')
ax.set_ylabel('vertical axis index')
ax.set_title('Vertical Cross-Section (Along-Track)')
pp = plt.imshow(subset2, cmap=plt.cm.Spectral_r)

# Add a colorbar to the bottom of the plot.
fig.subplots_adjust(bottom=0.15, left=0.06, right=0.94)
cbar_ax = fig.add_axes([0.12, 0.11, 0.76, 0.01])  
cbar = plt.colorbar(pp, cax=cbar_ax, orientation='horizontal')
# cbar.set_label(label=cross.attrs.get('units').decode('utf-8'),size=10)
#%%
# CROSS-TRACK VERTICAL PROFILE
# Choose the range of the subset, e.g.:
#   One complete row in the along-track dimension that slices through the eye of the storm
#   The lowest 76 elements in the vertical dimension
mysub3 = cross[start:end,:,50:175]
mysub3[mysub3==-9999.9]=0
# Transpose the array so the cross-track dimension will be the X-Axis
#   and the vertical dimension will be the Y-Axis
subset3 = np.transpose(mysub3)

# Draw the CROSS-TRACK VERTICAL PROFILE
fig = plt.figure(figsize=(4,9))
ax = plt.axes()
ax.set_xlabel('cross-track axis index')
ax.set_ylabel('vertical axis index')
ax.set_title('Vertical Cross-Section of the eye of the hurricane (Cross-Track)')
pp = plt.imshow(subset3, cmap=plt.cm.Spectral_r)

# Add a colorbar to the bottom of the plot.
fig.subplots_adjust(bottom=0.15,left=0.06, right=0.94)
cbar_ax = fig.add_axes([0.12, 0.11, 0.76, 0.015])  
cbar = plt.colorbar(pp, cax=cbar_ax, orientation='horizontal')
# cbar.set_label(label=cross.attrs.get('units').decode('utf-8'),size=10)


#%%
plt.imshow(gmidata[:,:,87])
# %%
#%% Plotting near surface
# Subset the 2-Dimensional variable over the eye of Hurricane Harvey
# Choose the range of the subset, e.g.:
#   170 rows in the along-track dimension,
#   All 49 elements in the cross-track dimension,
start = 2500
end = 2700
# mysub = prnsdata[start:end,:]
mysub = gmidata[:,:,70]
mylon = lonb[:,:]
mylat = latb[:,:]

# Draw the subset of near-surface precipitation rate 
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-100,-85,20,35])
plt.title('Total Precipitation Rate from GMI, 25 August 2017')

# Add coastlines and gridlines
ax.coastlines(resolution="50m",linewidth=0.5)
gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.8,color='#555555',alpha=0.5,linestyle='--')

# Axis labels
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = True

# Plot the scatter diagram 
pp = plt.scatter(mylon, mylat, c=mysub, cmap=plt.cm.Spectral, transform=ccrs.PlateCarree())

# Add a colorbar to the bottom of the plot.
fig.subplots_adjust(bottom=0.15,left=0.06,right=0.94)
cbar_ax = fig.add_axes([0.12, 0.11, 0.76, 0.025])  
cbar = plt.colorbar(pp,cax=cbar_ax,orientation='horizontal')
cbar.set_label(label=gmi.attrs.get('units').decode('utf-8'),size=10)
# %%

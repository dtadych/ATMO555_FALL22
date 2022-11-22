
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
datadirectory = 'Data/GPM_IMERG/'

# %% gotta make a list of names
Early = [
        "3B-DAY-E.MS.MRG.3IMERG.20170823-S000000-E235959.V06.nc4",
        "3B-DAY-E.MS.MRG.3IMERG.20170824-S000000-E235959.V06.nc4",
        "3B-DAY-E.MS.MRG.3IMERG.20170825-S000000-E235959.V06.nc4",
        "3B-DAY-E.MS.MRG.3IMERG.20170826-S000000-E235959.V06.nc4",
        "3B-DAY-E.MS.MRG.3IMERG.20170827-S000000-E235959.V06.nc4"]
Late = ["3B-DAY-L.MS.MRG.3IMERG.20170823-S000000-E235959.V06.nc4",
        "3B-DAY-L.MS.MRG.3IMERG.20170824-S000000-E235959.V06.nc4",
        "3B-DAY-L.MS.MRG.3IMERG.20170825-S000000-E235959.V06.nc4",
        "3B-DAY-L.MS.MRG.3IMERG.20170826-S000000-E235959.V06.nc4",
        "3B-DAY-L.MS.MRG.3IMERG.20170827-S000000-E235959.V06.nc4"]
Final = ["3B-DAY.MS.MRG.3IMERG.20170823-S000000-E235959.V06.nc4",
        "3B-DAY.MS.MRG.3IMERG.20170824-S000000-E235959.V06.nc4",
        "3B-DAY.MS.MRG.3IMERG.20170825-S000000-E235959.V06.nc4",
        "3B-DAY.MS.MRG.3IMERG.20170826-S000000-E235959.V06.nc4",
        "3B-DAY.MS.MRG.3IMERG.20170827-S000000-E235959.V06.nc4"]
# %% --- Reading in the data ---
data_early = xr.open_dataset(datadirectory+str(Early[0]))
data_early

data_late = xr.open_dataset(datadirectory+str(Late[0]))
data_late

data_final = xr.open_dataset(datadirectory+str(Final[0]))
data_final
# %% This makes them all come together
for i in Early:
    f = xr.open_dataset(datadirectory+i)
    data_early = xr.merge([data_early,f])

for i in Late:
    f = xr.open_dataset(datadirectory+i)
    data_late = xr.merge([data_late,f])

for i in Final:
    f = xr.open_dataset(datadirectory+i)
    data_final = xr.merge([data_final,f])

print(data_early)

# %% Slicing the dat
data = data_early
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
time = data.variables['time'][:]
earlyprecip = data['precipitationCal']
lateprecip = data_late['precipitationCal']
finalprecip = data_final['precipitationCal']

# print(earlyprecip)
#%%
earlyprecip[0].plot()

# %%
global_earlymean = earlyprecip.sum(("time"))
global_latemean = lateprecip.sum(("time"))
global_finalmean = finalprecip.sum(("time"))

# %% Trying to get the max precip for the storm

# lat = 29 lon -95
latitude = global_earlymean['lat'].values[1198]
longitude = global_earlymean['lon'].values[850]

print("The current latitude is ", latitude, 'and longitude is', longitude)
#%%
early_one_point = global_earlymean.sel(lat = latitude, lon = longitude)
print('max for early =', early_one_point.max())
late_one_point = global_latemean.sel(lat = latitude, lon = longitude)
print('max for late =', late_one_point.max())
final_one_point = global_finalmean.sel(lat = latitude, lon = longitude)
print('max for final =', final_one_point.max())


#%%
fig = plt.figure(figsize=(10,15))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
global_earlymean.plot(ax = ax)


#%%
# Bounding the area of interest
lat_min = -67
lat_max = -100
lon_min = 20
lon_max = 40

#%%
#Early Plot
fig, ax = plt.subplots(figsize = (7,5))
# fig = plt.figure(figsize=(10,15))
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.add_feature(cfeature.COASTLINE)
# test2=global_earlymean.transpose()
global_earlymean.plot(ax=ax)
# test2.plot(ax = ax)
# rotate world
# plotty = global_earlymean.rotate(90, origin = (0,0), use_radians=False)

# ax.gridlines(draw_labels=False,
                # xlocs=np.arrange(-180,180,1.),
                # ylocs=np.arrange(-90,90,1.),
                # linewidth=1, color='k', alpha=0.5,linestyle=':')

# plotty.plot(ax=ax, edgecolor='grey', linewidth=1, alpha=1)
ylim = (lat_max,lat_min)
xlim = (lon_max, lon_min)

ax.set_xlim(xlim)
ax.set_ylim(ylim)
# didn't work
# color = ax.pcolormesh(global_earlymean.lon,global_earlymean.lat,global_earlymean,cmap='viridis')
# cbar = fig.colorbar(color, ax=ax, orientation = 'horizontal')
ax_color=ax.imshow(np.flip(global_earlymean.transpose(), axis = 0),cmap='viridis')
# ax.coastlines(color='black')
cbar = plt.colorbar(ax_color)
cbar.set_label("Total Precip (mm)")

# ax.add_feature(cfeature.COASTLINE)
ax.set_title('Hurricane Harvey \n(IMERG Total Early for 8/23-8/27)', fontsize = 14)
# ax.set_axis_off()
plt.show()
# plt.savefig("Tadych_Assignment8_HurricaneHarvey_early")
#%% Plot from Mohammed
fig=plt.figure(figsize=(20,20))
ax
#%% Late Plot
plotty=global_latemean

fig, ax = plt.subplots(figsize = (7,5))
# fig = plt.figure(figsize=(10,15))
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.add_feature(cfeature.COASTLINE)
plotty.plot(ax=ax)
ylim = (lat_max,lat_min)
xlim = (lon_max, lon_min)

ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax_color=ax.imshow(np.flip(plotty.transpose(), axis = 0),cmap='viridis')
cbar = plt.colorbar(ax_color)
cbar.set_label("Total Precip (mm)")


ax.set_title('Hurricane Harvey \n(IMERG Total Late for 8/23-8/27)', fontsize = 14)
# ax.set_axis_off()
plt.show()
# plt.savefig("Tadych_Assignment8_HurricaneHarvey_late")


#%% Final Plot
plotty=global_finalmean

fig, ax = plt.subplots(figsize = (7,5))
plotty.plot(ax=ax)
ylim = (lat_max,lat_min)
xlim = (lon_max, lon_min)

ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax_color=ax.imshow(np.flip(plotty.transpose(), axis = 0),cmap='viridis')
cbar = plt.colorbar(ax_color)
cbar.set_label("Total Precip (mm)")


ax.set_title('Hurricane Harvey \n(IMERG Final for 8/23-8/27)', fontsize = 14)
# ax.set_axis_off()
plt.show()
# plt.savefig("Tadych_Assignment8_HurricaneHarvey_Final")
# %% Differences
early_late_dif = global_earlymean - global_latemean
early_final_dif = global_earlymean - global_finalmean
late_final_dif = global_latemean - global_finalmean

plotty=early_late_dif

fig, ax = plt.subplots(figsize = (7,5))
plotty.plot(ax=ax, cmap='RdBu')
ylim = (lat_max,lat_min)
xlim = (lon_max, lon_min)

ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax_color=ax.imshow(np.flip(plotty.transpose(), axis = 0),cmap='RdBu')
cbar = plt.colorbar(ax_color)
cbar.set_label("Precipitation (mm)")


ax.set_title('Hurricane Harvey \n(IMERG Early-Late Differences for 8/23-8/27)', fontsize = 14)
# ax.set_axis_off()
plt.show()
# plt.savefig("Tadych_Assignment8_HurricaneHarvey_Final")
# %%
plotty=early_final_dif

fig, ax = plt.subplots(figsize = (7,5))
plotty.plot(ax=ax, cmap='RdBu')
ylim = (lat_max,lat_min)
xlim = (lon_max, lon_min)

ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax_color=ax.imshow(np.flip(plotty.transpose(), axis = 0),cmap='RdBu')
cbar = plt.colorbar(ax_color)
cbar.set_label("Precipitation (mm)")


ax.set_title('Hurricane Harvey \n(IMERG Early-Final Differences for 8/23-8/27)', fontsize = 14)
# ax.set_axis_off()
plt.show()
# plt.savefig("Tadych_Assignment8_HurricaneHarvey_Final")

# %%
plotty=late_final_dif

fig, ax = plt.subplots(figsize = (7,5))
plotty.plot(ax=ax, cmap='RdBu')
ylim = (lat_max,lat_min)
xlim = (lon_max, lon_min)

ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax_color=ax.imshow(np.flip(plotty.transpose(), axis = 0),cmap='RdBu')
cbar = plt.colorbar(ax_color)
cbar.set_label("Precipitation (mm)")


ax.set_title('Hurricane Harvey \n(IMERG Late-Final Differences for 8/23-8/27)', fontsize = 14)
# ax.set_axis_off()
plt.show()
# plt.savefig("Tadych_Assignment8_HurricaneHarvey_Final")

# %% Creating a Scatter Plot
# da.where(da.y < 2)

# Bounding the area of interest
# lat_min = -85
# lat_max = -100
# lon_min = 20
# lon_max = 35
test = global_earlymean.where(global_earlymean.lon >= lon_min)

#%%
da = global_earlymean
da2 = da.isel(lon=slice(20,35),lat=slice(-100,-85))
da2

lc = da.coords["lon"]

la = da.coords["lat"]

da2 = da.loc[
         dict(lon=lc[(lc > -100) & (lc < -85)], lat=la[(la > 20) & (la < 35)])
                         ]

early = da2

da = global_latemean
da2 = da.isel(lon=slice(20,35),lat=slice(-100,-85))
da2

lc = da.coords["lon"]

la = da.coords["lat"]

da2 = da.loc[
         dict(lon=lc[(lc > -100) & (lc < -85)], lat=la[(la > 20) & (la < 35)])
                         ]

late = da2

da = global_finalmean
da2 = da.isel(lon=slice(20,35),lat=slice(-100,-85))
da2

lc = da.coords["lon"]

la = da.coords["lat"]

da2 = da.loc[
         dict(lon=lc[(lc > -100) & (lc < -85)], lat=la[(la > 20) & (la < 35)])
                         ]

final = da2

# %%
fig, ax = plt.subplots(figsize = (7,5))
ax.scatter(final,early,color='red',label='Early')
ax.scatter(final,late,color='blue',label='Late')
# Plotting the 1:1 line
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
# now plot both limits against eachother
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.xlabel('Final Precipitation Values (5mm/day)')
plt.ylabel('Precipitation (5mm/day)')
ax.legend()

# %% Calculating statistics
fe_error = final - early
fl_error = final - late

febias = final.mean() - early.mean()
flbias = final.mean() - late.mean()
print('final-early bias is ', int(febias),' and final-late bias is ', int(flbias))

#%% RMSE
sq_fe_error = fe_error*fe_error
sq_fl_error = fl_error*fl_error
sq_fe_error
#%%
count = len(sq_fe_error)
count
#%%
rmse_fe = math.sqrt(sq_fe_error.sum()/count)
rmse_fl = math.sqrt(sq_fl_error.sum()/count)
print('final-early RMSE is ', int(rmse_fe),' and final-late RMSE is ', int(rmse_fl))

#%%
print('Pearson Correlation coefficient for final-early')
df1 = final
df2 = early
x=pd.DataFrame(df1)
y=pd.DataFrame(df2)
r = sp.pearsonr(x.all(),y.all())
print(r)

print('Pearson Correlation coefficient for final-late')
df1 = final
df2 = late
x=pd.DataFrame(df1)
y=pd.DataFrame(df2)
r = sp.pearsonr(x.all(),y.all())
print(r)
# print('  rsq = ',round(r*r,3))
# print('  pval = ',round(df1.corr(df2, method=pearsonr_pval),4))
#%% From the outputs
re = 0.7062488958007375
rl = 0.7208436724565759

rsq_e = re*re
rsq_l = rl*rl
print(rsq_e,'and',rsq_l)


#%%
print(np.var(early))

# %% -------- Part 3 ----------
datadirectory = 'Data/GPM_IMERG/IR_only/'

# gotta make a list of names
ironly = [
        "3B-HHR.MS.MRG.3IMERG.20170823-S000000-E002959.0000.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170823-S200000-E202959.1200.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170824-S200000-E202959.1200.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S000000-E002959.0000.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S003000-E005959.0030.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S010000-E012959.0060.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S013000-E015959.0090.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S020000-E022959.0120.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S023000-E025959.0150.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S030000-E032959.0180.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S033000-E035959.0210.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S040000-E042959.0240.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S043000-E045959.0270.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S050000-E052959.0300.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S053000-E055959.0330.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S060000-E062959.0360.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S063000-E065959.0390.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S070000-E072959.0420.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S073000-E075959.0450.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S080000-E082959.0480.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S083000-E085959.0510.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S090000-E092959.0540.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S093000-E095959.0570.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S100000-E102959.0600.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S103000-E105959.0630.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S110000-E112959.0660.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S113000-E115959.0690.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S120000-E122959.0720.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S123000-E125959.0750.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S130000-E132959.0780.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S133000-E135959.0810.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S140000-E142959.0840.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S143000-E145959.0870.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S150000-E152959.0900.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S153000-E155959.0930.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S160000-E162959.0960.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S163000-E165959.0990.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S170000-E172959.1020.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S173000-E175959.1050.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S180000-E182959.1080.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S183000-E185959.1110.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S190000-E192959.1140.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S193000-E195959.1170.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S200000-E202959.1200.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S203000-E205959.1230.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S210000-E212959.1260.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S213000-E215959.1290.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S220000-E222959.1320.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S223000-E225959.1350.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S230000-E232959.1380.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170825-S233000-E235959.1410.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170826-S000000-E002959.0000.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170826-S003000-E005959.0030.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170826-S010000-E012959.0060.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170826-S013000-E015959.0090.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170826-S200000-E202959.1200.V06B.HDF5.SUB.nc4"
        ,"3B-HHR.MS.MRG.3IMERG.20170827-S200000-E202959.1200.V06B.HDF5.SUB.nc4"]

# --- Reading in the data ---
IR_only = xr.open_dataset(datadirectory+str(ironly[0]))
IR_only

#%%
for i in ironly:
    f = xr.open_dataset(datadirectory+i)
    IR_only = xr.merge([IR_only,f])

IR_only
# %% Slicing the dat
data = IR_only
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
time = data.variables['time'][:]
IRprecip = data['IRprecipitation']

# print(earlyprecip)
#%%
IRprecip[4].plot()

# %%
Ir_sum = IRprecip.sum(("time"))
Ir_sum.plot()
#%%
da = Ir_sum
da2 = da.isel(lon=slice(20,35),lat=slice(-100,-85))
da2

lc = da.coords["lon"]

la = da.coords["lat"]

da2 = da.loc[
         dict(lon=lc[(lc > -100) & (lc < -85)], lat=la[(la > 20) & (la < 35)])
                         ]

IR = da2
IR.plot()
#%%
final.plot()

#%%
difference = final - IR
difference.plot()
# %% -------- Part 4 ----------
# The latitude of Houston, TX, USA is 29.749907, and the longitude is -95.358421

# latitude = global_earlymean['lat'].values[1198]
# longitude = global_earlymean['lon'].values[850]

# print("The current latitude is ", latitude, 'and longitude is', longitude)

latitude = data_final['precipitationCal']['lat'].values[1197]
longitude = data_final['precipitationCal']['lon'].values[846]

print("The current latitude is ", latitude, 'and longitude is', longitude)
#%%
one_point = finalprecip.sel(lat = latitude, lon = longitude)
one_point
#%%
datetimeindex = one_point.indexes['time'].to_datetimeindex()
#%%
one_point['time'] = datetimeindex
one_point

#%%
one_point.plot()

#%%
one_point_df = pd.DataFrame(one_point)
one_point_df
# %%
one_point_df.index = datetimeindex
# one_point_df[0]=one_point_df['Precipitation (mm)']
# %%
one_point_df['day'] = pd.DatetimeIndex(one_point_df.index).day
one_point_df['day'] = int(one_point_df['day'])
one_point_df
#%%
f, ax = plt.subplots(figsize=(7, 5))
# one_point.plot.line(hue='lat',
#                     marker="o",
#                     ax=ax,
#                     color="white",
#                     markerfacecolor="blue",
#                     markeredgecolor="blue"
#                     )
ax.scatter(one_point_df.day,one_point_df[0],color='blue')

ax.set(title="IMERG Precipitation Estimate Houston")
plt.xlabel('Day (August 2017)')
plt.ylabel('Preciptiation (mm)')
# %%

# %%

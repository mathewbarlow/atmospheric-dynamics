#
# run on python 3.7
#
# python code to plot 500hPa heights and geostrophic winds
# from real-time GFS analysis
#
# the date and lat-lon range can be set below
#
# (poorly) coded by Mathew Barlow, Mathew_Barlow@uml.edu
#
# Support from NSF AGS-1623912 and AGS-1657921 is gratefully acknowledged
#
# version 1.0:  5 Dec 2021
# version 1.1:  6 Dec 2021, fixed missing parentheses after plt.show
#

# for numerical calculations
import numpy as np

# for general plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# for map projections and display
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# for time-related calculations
from datetime import datetime

# for accessing data online
import xarray as xr


# VALUES TO SET *************************************************
# set date, lat-lon range, and level (in hPa)
mydate = '20211207'
myhour = '12'
(lon1, lon2) = (-140, -60)
(lat1, lat2) = (15, 60)
level = 500
# ****************************************************************

# convert lon to 0-360
lon1 = lon1 + 360
lon2 = lon2 + 360

# some useful constants
re = 6.37e6
g = 9.81
cp = 1004.5
r = 2.0*cp/7.0
kap = r/cp
omega = 7.292e-5


# open dataset and retreive variables
url = 'https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/' + \
    'Global_0p25deg_ana/GFS_Global_0p25deg_ana_' + mydate + '_' + \
    myhour + '00.grib2'

ds = xr.open_dataset(url)
# to list all variables:  print(list(ds.variables))

lats = ds['lat'].values
lons = ds['lon'].values
levs = ds['isobaric'].values/100

iz = np.argmin(np.abs(levs - level))


# get array indices for lat-lon range
# specified above
iy1 = np.argmin(np.abs(lats - lat1))
iy2 = np.argmin(np.abs(lats - lat2))
ix1 = np.argmin(np.abs(lons - lon1))
ix2 = np.argmin(np.abs(lons - lon2))

nlev = levs.size
nx = lons.size
ny = lats.size

lats = ds['lat'][iy2:iy1].values
lons = ds['lon'][ix1:ix2].values

t = ds['Temperature_isobaric'][0, iz, iy2:iy1, ix1:ix2].values
u = ds['u-component_of_wind_isobaric'][0, iz, iy2:iy1, ix1:ix2].values
v = ds['v-component_of_wind_isobaric'][0, iz, iy2:iy1, ix1:ix2].values
hgt = ds['Geopotential_height_isobaric'][0, iz, iy2:iy1, ix1:ix2].values
absvort = ds['Absolute_vorticity_isobaric'][0, iz, iy2:iy1, ix1:ix2].values

# make latitude run from south to north
lats = np.flip(lats, axis=0)
t = np.flip(t, axis=0)
u = np.flip(u, axis=0)
v = np.flip(v, axis=0)
hgt = np.flip(hgt, axis=0)
absvort = np.flip(absvort, axis=0)

# define 2d lon and lat, to match variables
lons2d, lats2d = np.meshgrid(lons, lats)

# define Coriolis parameter
f = 2.0*omega*np.sin(lats2d*np.pi/180.0)
f0 = 2.0*omega*np.sin(45*np.pi/180.0)

# relative vorticity
relvort = absvort - f


# define derivative functions
def ddx(var):
    # use center-difference,
    # except for side boundaries, where use poly interp
    x = (re*np.cos(lats2d*np.pi/180)*np.pi/180)*lons2d
    ddx_var = np.zeros_like(var)
    ddx_var[:, 1:-1] = (var[:, 2:] - var[:, 0:-2]) / (x[:, 2:] - x[:, 0:-2])

    ddx_var[:, 0] = ((- 3.0*var[:, 0] + 4.0*var[:, 1] - var[:, 2]) /
                     (- 3.0*x[:, 0] + 4.0*x[:, 1] - x[:, 2]))
    ddx_var[:, -1] = ((- 3.0*var[:, -1] + 4.0*var[:, -2] - var[:, -3]) /
                      (- 3.0*x[:, -1] + 4.0*x[:, -2] - x[:, -3]))

    return(ddx_var)


def ddy(var):
    # use center-difference,
    # except for N/S boundaries, where use poly interp
    y = (re*np.pi/180)*lats2d
    ddy_var = np.zeros_like(var)
    ddy_var[1:-1, :] = (var[2:, :] - var[0:-2, :]) / (y[2:, :] - y[0:-2, :])

    ddy_var[0, :] = ((- 3.0*var[0, :] + 4.0*var[1, :] - var[2, :]) /
                     (- 3.0*y[0, :] + 4.0*y[1, :] - y[2, :]))
    ddy_var[-1, :] = ((- 3.0*var[-1, :] + 4.0*var[-2, :] - var[-3, :]) /
                      (- 3.0*y[-1, :] + 4.0*y[-2, :] - y[-3, :]))

    return(ddy_var)


# define simple smoother
# scipy has much more sophisticated smoothers
def smooth_121(f_in, niter):
    # 1-2-1 smoother in x and y
    f_out = np.copy(f_in)
    iter = 0
    while(iter <= niter):
        f_sx = (np.roll(f_out, -1, axis=1)+2*f_out+np.roll(f_out, 1, axis=1))/4
        f_sy = (np.roll(f_out, -1, axis=0)+2*f_out+np.roll(f_out, 1, axis=0))/4
        f_out[1:-1, 1:-1] = (f_sx[1:-1, 1:-1]+f_sy[1:-1, 1:-1])/2
        f_out[0, 1:-1] = f_sx[0, 1:-1]
        f_out[-1, 1:-1] = f_sx[-1, 1:-1]
        f_out[1:-1, 0] = f_sy[1:-1, 0]
        f_out[1:-1, -1] = f_sy[1:-1, -1]
        iter = iter+1
    return(f_out)


# function for spatial correlation
def scorr(a, b):
    abar = np.mean(a)
    bbar = np.mean(b)
    covar = np.sum((a-abar)*(b-bbar))
    avar = np.sum((a-abar)**2)
    bvar = np.sum((b-bbar)**2)
    r = covar/np.sqrt(avar*bvar)
    return(r)


# calculate smoothed hgt field
# note that GFS resolution is sufficient to cause problems
# when calculating geostrophic wind from the unsmoothed heights
hgts = smooth_121(hgt, 60)

ug = (-g/f)*ddy(hgts)
vg = (g/f)*ddx(hgts)


# now make some plots

# get date for plotting
fdate = datetime.strptime(mydate, '%Y%m%d').strftime('%d %b %Y')

plt.close("all")


plt.figure(1)

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon1-0.01, lon2+0.01, lat1-0.01, lat2+0.01],
              crs=ccrs.PlateCarree())

clevs = np.arange(4900, 6000, 100)

plt.contour(lons, lats, hgts, clevs, transform=ccrs.PlateCarree(),
            colors='black', linewidths=0.5)

sk = 15
plt.quiver(lons[::sk], lats[::sk], ug[::sk, ::sk], vg[::sk, ::sk],
           transform=ccrs.PlateCarree(), scale=1e3, color='black')

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                  linestyle='dashed', alpha=0.5)

ax.coastlines(linewidth=0.8, color='gray', alpha=0.5)

gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator(np.arange(lon1-360, lon2-360+10, 10))
gl.ylocator = mticker.FixedLocator(np.arange(lat1, lat2+5, 5))
plt.title(format(level,'g')+'hPa SMOOTH HGT, GEO WINDS\n'+myhour+'Z '+fdate)


plt.figure(2)

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon1-0.01, lon2+0.01, lat1-0.01, lat2+0.01],
              crs=ccrs.PlateCarree())


plt.contour(lons, lats, hgt, clevs, transform=ccrs.PlateCarree(),
            colors='black', linewidths=0.5)

sk = 15
plt.quiver(lons[::sk], lats[::sk], u[::sk, ::sk], v[::sk, ::sk],
           transform=ccrs.PlateCarree(), scale=1e3, color='black')

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                  linestyle='dashed', alpha=0.5)

ax.coastlines(linewidth=0.8, color='gray', alpha=0.5)

gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator(np.arange(lon1-360, lon2-360+10, 10))
gl.ylocator = mticker.FixedLocator(np.arange(lat1, lat2+5, 5))
plt.title(format(level,'g')+'hPa HGT, WINDS\n'+myhour+'Z '+fdate)


plt.show()

"""
Diagnostic plots for the meteorological data pipeline.

Only invoked when a diagnostic-plot flag is set in the model setup.
"""

import numpy as np

MODULE_NAME = "monarchs.met_data.diagnostics"


def generate_met_dem_diagnostic_plots(old_era5_grid, era5_grid, ilats, ilons, iheights):
    """
    Generate diagnostic plots to visualise the regridding of ERA5 data onto the
    DEM mesh.

    Parameters
    ----------
    old_era5_grid
    era5_grid
    ilats
    ilons
    iheights

    Returns
    -------

    """
    import cartopy.crs as ccrs
    import cartopy
    from matplotlib import pyplot as plt

    routine_name = "generate_met_dem_diagnostic_plots"
    print(f"{MODULE_NAME}.{routine_name}: Generating diagnostic plots")
    latmax = ilats.max()
    latmin = ilats.min()
    lonmax = ilons.max()
    lonmin = ilons.min()
    fig1 = plt.figure()
    cmap = "viridis"
    projection = ccrs.PlateCarree()
    ax1 = fig1.add_subplot(211, projection=projection)
    ax1.set_title("Original vs regridded met data")
    ax1.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.gridlines(draw_labels=True)
    lons = old_era5_grid["long"][:]
    lats = old_era5_grid["lat"][:]
    temperature = old_era5_grid["temperature"][:]
    vmin = np.min(temperature[0])
    vmax = np.max(temperature[0])
    levels = np.linspace(vmin, vmax, 20)
    ax1.contourf(
        lons,
        lats,
        temperature[0],
        cmap=cmap,
        transform=projection,
        levels=levels,
        vmin=vmin,
        vmax=vmax,
    )
    ax2 = fig1.add_subplot(212, projection=projection)
    ax2.coastlines()
    ax2.gridlines(draw_labels=True)
    ax2.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    ax2.add_feature(cartopy.feature.LAND)
    cont = ax2.contourf(
        ilons,
        ilats,
        era5_grid["temperature"][0],
        cmap=cmap,
        transform=projection,
        levels=levels,
        vmin=vmin,
        vmax=vmax,
    )
    cbar_ax = fig1.add_axes((0.85, 0.15, 0.05, 0.7))
    fig1.colorbar(cont, ticks=levels, extend="both", cax=cbar_ax)
    fig2 = plt.figure()
    ax3 = fig2.add_subplot(111, projection=projection)
    ax3.coastlines()
    ax3.gridlines(draw_labels=True)
    cont = ax3.contourf(
        ilons,
        ilats,
        iheights,
        cmap=cmap,
        levels=20,
        transform=ccrs.PlateCarree(),
    )
    ax3.title.set_text("Initial DEM height profile")
    fig2.colorbar(cont, extend="both")
    plt.show()

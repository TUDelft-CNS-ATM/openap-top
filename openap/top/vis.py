import numpy as np
from openap import aero
from cartopy import crs as ccrs
from cartopy.feature import OCEAN, LAND, BORDERS
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def trajectory_on_map(df, windfield=None, ax=None, wind_sample=4):

    lat1, lon1 = df.lat.iloc[0], df.lon.iloc[0]
    lat2, lon2 = df.lat.iloc[-1], df.lon.iloc[-1]

    latmin, latmax = min(lat1, lat2), max(lat1, lat2)
    lonmin, lonmax = min(lon1, lon2), max(lon1, lon2)

    if ax is None:
        ax = plt.axes(
            projection=ccrs.TransverseMercator(
                central_longitude=df.lon.mean(), central_latitude=df.lat.mean()
            )
        )

    ax.set_extent([lonmin - 4, lonmax + 4, latmin - 2, latmax + 2])
    ax.add_feature(OCEAN, facecolor="#d1e0e0", zorder=-1, lw=0)
    ax.add_feature(LAND, facecolor="#f5f5f5", lw=0)
    ax.add_feature(BORDERS, lw=0.5, color="gray")
    ax.gridlines(draw_labels=True, color="gray", alpha=0.5, ls="--")
    ax.coastlines(resolution="50m", lw=0.5, color="gray")

    if windfield is not None:
        # get the closed altitude
        h_max = df.alt.max() * aero.ft
        fl = int(round(h_max / aero.ft / 100, -1))
        idx = np.argmin(abs(windfield.h.unique() - h_max))
        df_wind = (
            windfield.query(f"h=={windfield.h.unique()[idx]}")
            .query(f"longitude <= {lonmax + 2}")
            .query(f"longitude >= {lonmin - 2}")
            .query(f"latitude <= {latmax + 2}")
            .query(f"latitude >= {latmin - 2}")
        )

        ax.barbs(
            df_wind.longitude.values[::wind_sample],
            df_wind.latitude.values[::wind_sample],
            df_wind.u.values[::wind_sample],
            df_wind.v.values[::wind_sample],
            transform=ccrs.PlateCarree(),
            color="k",
            length=5,
            lw=0.5,
            label=f"Wind FL{fl}",
        )

    # great circle
    ax.scatter(lon1, lat1, c="darkgreen", transform=ccrs.Geodetic())
    ax.scatter(lon2, lat2, c="tab:red", transform=ccrs.Geodetic())

    ax.plot(
        [lon1, lon2],
        [lat1, lat2],
        label="Great Circle",
        color="tab:red",
        ls="--",
        transform=ccrs.Geodetic(),
    )

    # trajectory
    ax.plot(
        df.lon,
        df.lat,
        color="tab:green",
        transform=ccrs.Geodetic(),
        linewidth=3,
        marker="o",
        label="Optimal",
    )

    ax.legend()

    return plt
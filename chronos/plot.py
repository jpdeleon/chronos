# -*- coding: utf-8 -*-

r"""
classes for plotting cluster properties
"""
# Import standard library
import logging
import itertools

# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from astropy.coordinates import Angle, SkyCoord, Distance
import astropy.units as u
from astropy.timeseries import LombScargle
from mpl_toolkits.mplot3d import Axes3D

# Import from package
from chronos.search import ClusterCatalog
from chronos.utils import get_transformed_coord, get_tois, get_mamajek_table

log = logging.getLogger(__name__)

__all__ = ["plot_rdp_pmrv", "plot_xyz_uvw", "plot_hrd", "plot_tls", "plot_hrd_spectral_types"]

def plot_lomb_scargle(time, flux, flux_err, figsize=(8,8)):
    """
    """
    frequency, power = LombScargle(time, flux, flux_err).autopower(minimum_frequency=0.05,
                                                                   #maximum_frequency=2.0
                                                                  )
    best_frequency = frequency[np.argmax(power)]
    t_fit = np.linspace(0, 1)
    ls = LombScargle(time, flux, flux_err)
    y_fit = ls.model(t_fit, best_frequency)

    fig, ax = pl.subplots(2, 1, figsize=figsize)
    ax[0].plot(1./frequency, power)

    ax[1].plot(time/per%1-0.5, flux, '.')
    ax[1].plot(t_fit-0.5, y_fit)
    return fig

def plot_tls(results, **kwargs):
    """

    Attributes
    ----------
    results : dict
        results of after running tls.power()
    * kwargs : dict
        plotting kwargs e.g. {'figsize': (8,8), 'constrained_layout': True}

    Returns
    -------
    fig : figure object
    """
    fig, ax = pl.subplots(2, 1, **kwargs)

    n=0
    label=f'TLS={results.period:.3}'
    ax[n].axvline(results.period, alpha=0.4, lw=3, label=label)
    ax[n].set_xlim(np.min(results.periods), np.max(results.periods))

    for i in range(2, 10):
        ax[n].axvline(i*results.period, alpha=0.4, lw=1, linestyle="dashed")
        ax[n].axvline(results.period / i, alpha=0.4, lw=1, linestyle="dashed")
    ax[n].set_ylabel(r'SDE')
    ax[n].set_xlabel('Period (days)')
    ax[n].plot(results.periods, results.power, color='black', lw=0.5)
    ax[n].set_xlim(0, max(results.periods))

    n=1
    ax[n].plot(results.model_folded_phase, results.model_folded_model, color='red')
    ax[n].scatter(results.folded_phase, results.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
    ax[n].set_xlabel('Phase')
    ax[n].set_ylabel('Relative flux');
    return fig

def plot_rdp_pmrv(
    df, target_gaia_id=None, match_id=True, df_target=None, target_label=None, figsize=(10, 10)
):
    """
    Plot ICRS position and proper motions in 2D scatter plots,
    and parallax and radial velocity in kernel density

    Parameters
    ----------
    df : pandas.DataFrame
        contains ra, dec, parallax, pmra, pmdec, radial_velocity columns
    target_gaiaid : int
        target gaia DR2 id
    """
    assert len(df) > 0, "df is empty"
    fig, axs = pl.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax = axs.flatten()

    n = 0
    x, y = "ra", "dec"
    # df.plot.scatter(x=x, y=y, ax=ax[n])
    ax[n].scatter(df[x], df[y], marker="o")
    if target_gaia_id is not None:
        idx = df.source_id.astype(int).isin([target_gaia_id])
        if match_id:
            errmsg = (
                f"Given cluster does not contain the target gaia id [{target_gaia_id}]"
            )
            assert sum(idx) > 0, errmsg
            ax[n].plot(
                df.loc[idx, x],
                df.loc[idx, y],
                marker=r"$\star$",
                c="y",
                ms="25",
                label=target_label,
            )
        else:
            assert df_target is not None, "provide df_target"
            ax[n].plot(
                df_target[x],
                df_target[y],
                marker=r"$\star$",
                c="y",
                ms="25",
                label=target_label,
            )
    ax[n].set_xlabel("R.A. [deg]")
    ax[n].set_ylabel("Dec. [deg]")
    text = len(df[["ra", "dec"]].dropna())
    ax[n].text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes)
    if target_label is not None:
        ax[n].legend(loc="best")
    n = 1
    par = "parallax"
    df[par].plot.kde(ax=ax[n])
    if target_gaia_id is not None:
        idx = df.source_id.astype(int).isin([target_gaia_id])
        if match_id:
            errmsg = (
                f"Given cluster does not contain the target gaia id [{target_gaia_id}]"
            )
            assert sum(idx) > 0, errmsg
            ax[n].axvline(
                df.loc[idx, par].values[0], 0, 1, c="k", ls="--", label=target_label
            )
        else:
            assert df_target is not None, "provide df_target"
            ax[n].axvline(df_target[par], 0, 1, c="k", ls="--", label=target_label)

        if target_label is not None:
            ax[n].legend(loc="best")
    ax[n].set_xlabel("Parallax [mas]")
    text = len(df[par].dropna())
    ax[n].text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes)
    n = 2
    x, y = "pmra", "pmdec"
    # df.plot.scatter(x=x, y=y, ax=ax[n])
    ax[n].scatter(df[x], df[y], marker="o")
    if target_gaia_id is not None:
        idx = df.source_id.astype(int).isin([target_gaia_id])
        if match_id:
            errmsg = (
                f"Given cluster does not contain the target gaia id [{target_gaia_id}]"
            )
            assert sum(idx) > 0, errmsg
            ax[n].plot(
                df.loc[idx, x], df.loc[idx, y], marker=r"$\star$", c="y", ms="25"
            )
        else:
            assert df_target is not None, "provide df_target"
            ax[n].plot(df_target[x], df_target[y], marker=r"$\star$", c="y", ms="25")
    ax[n].set_xlabel("PM R.A. [deg]")
    ax[n].set_ylabel("PM Dec. [deg]")
    text = len(df[["pmra", "pmdec"]].dropna())
    ax[n].text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes)
    n = 3
    par = "radial_velocity"
    try:
        df[par].plot.kde(ax=ax[n])
        if target_gaia_id is not None:
            idx = df.source_id.astype(int).isin([target_gaia_id])
            if match_id:
                errmsg = f"Given cluster does not contain the target gaia id [{target_gaia_id}]"
                assert sum(idx) > 0, errmsg
                ax[n].axvline(
                    df.loc[idx, par].values[0], 0, 1, c="k", ls="--", label=target_label
                )
            else:
                ax[n].axvline(df_target[par], 0, 1, c="k", ls="--", label=target_label)
        ax[n].set_xlabel("RV [km/s]")
        text = len(df[par].dropna())
        ax[n].text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes)
    except Exception as e:
        print(e)
        # catalog_name = df.Cluster.unique()()
        raise ValueError(f"radial_velocity is not available")  # in {catalog_name}
    return fig


def plot_xyz_uvw(df, target_gaia_id=None, match_id=True, df_target=None, verbose=True, figsize=(12, 8)):
    """
    Plot 3D position in galactocentric (xyz) frame
    and proper motion with radial velocity in galactic cartesian velocities
    (UVW) frame

    Parameters
    ----------
    df : pandas.DataFrame
        contains ra, dec, parallax, pmra, pmdec, radial_velocity columns
    target_gaiaid : int
        target gaia DR2 id
    df_target : pandas.Series
        target's gaia parameters

    Note: U is positive towards the direction of the Galactic center (GC);
    V is positive for a star with the same rotational direction as the Sun going around the galaxy,
    with 0 at the same rotation as sources at the Sun’s distance,
    and W positive towards the north Galactic pole

    U,V,W can be converted to Local Standard of Rest (LSR) by subtracting V = 238 km/s,
    the adopted rotation velocity at the position of the Sun from Marchetti et al. (2018).
    """
    assert len(df) > 0, "df is empty"
    fig, axs = pl.subplots(2, 3, figsize=figsize, constrained_layout=True)
    ax = axs.flatten()

    if not np.all(df.columns.isin("X Y Z U V W".split())):
        df = get_transformed_coord(df, frame="galactocentric", verbose=verbose)

    n = 0
    for (i, j) in itertools.combinations(["X", "Y", "Z"], r=2):
        if target_gaia_id is not None:
            idx = df.source_id.astype(int).isin([target_gaia_id])
            if match_id:
                errmsg = f"Given cluster does not contain the target gaia id [{target_gaia_id}]"
                assert sum(idx) > 0, errmsg
                ax[n].plot(
                    df.loc[idx, i], df.loc[idx, j], marker=r"$\star$", c="y", ms="25"
                )
            else:
                assert df_target is not None, "provide df_target"
                ax[n].plot(
                    df_target[i], df_target[j], marker=r"$\star$", c="y", ms="25"
                )
        # df.plot.scatter(x=i, y=j, ax=ax[n])
        ax[n].scatter(df[i], df[j], marker="o")
        ax[n].set_xlabel(i + " [pc]")
        ax[n].set_ylabel(j + " [pc]")
        text = len(df[[i, j]].dropna())
        ax[n].text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes)
        n += 1

    n = 3
    for (i, j) in itertools.combinations(["U", "V", "W"], r=2):
        if target_gaia_id is not None:
            idx = df.source_id.astype(int).isin([target_gaia_id])
            if match_id:
                errmsg = f"Given cluster does not contain the target gaia id [{target_gaia_id}]"
                assert sum(idx) > 0, errmsg
                ax[n].plot(
                    df.loc[idx, i], df.loc[idx, j], marker=r"$\star$", c="y", ms="25"
                )
            else:
                ax[n].plot(
                    df_target[i], df_target[j], marker=r"$\star$", c="y", ms="25"
                )
        # df.plot.scatter(x=i, y=j, ax=ax[n])
        ax[n].scatter(df[i], df[j], marker="o")
        ax[n].set_xlabel(i + " [km/s]")
        ax[n].set_ylabel(j + " [km/s]")
        text = len(df[[i, j]].dropna())
        ax[n].text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes)
        n += 1

    return fig


def plot_hrd(
    df,
    target_gaia_id=None,
    match_id=True,
    df_target=None,
    target_label=None,
    figsize=(8, 8),
    yaxis='abs_gmag',
    xaxis='bp_rp',
    ax=None
):
    """Plot HR Diagram with absolute magnitude and color from Gaia photometry

    Parameters
    ----------
    df : pd.DataFrame
        cluster memeber properties
    match_id : bool
        checks if target gaiaid in df
    df_target : pd.Series
        info of target
    xaxis, yaxis : str
        parameter to plot
    ax : axis

    Notes:
    M_G vs. Bp-Rp is supported, later try Teff vs color
    and superpose spec types from mamajek table
    """
    assert len(df) > 0, "df is empty"
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=figsize, constrained_layout=True)
    if target_gaia_id is not None:
        idx = df.source_id.astype(int).isin([target_gaia_id])
        if match_id:
            errmsg = f"Given cluster catalog does not contain the target gaia id [{target_gaia_id}]"
            assert sum(idx) > 0, errmsg
            ax.plot(
                x=df.loc[idx, xaxis],
                y=df.loc[idx, yaxis],
                marker=r"$\star$",
                c="y",
                ms="25",
                label=target_label,
            )
        else:
            assert df_target is not None, "provide df_target"
            df_target["distance"] = Distance(parallax=df_target["parallax"] * u.mas).pc
            if yaxis=="abs_gmag":
                df_target["abs_gmag"] = (
                    df_target["phot_g_mean_mag"]
                    - 5.0 * (np.log10(df_target["distance"]))
                    - 1
                )
            ax.plot(
                x=df_target[xaxis],
                y=df_target[yaxis],
                marker=r"$\star$",
                c="y",
                ms="25",
                label=target_label,
            )
        if target_label is not None:
            ax.legend(loc="best")
    # df.plot.scatter(ax=ax, x="bp_rp", y="abs_gmag", marker=".")
    ax.scatter(df[xaxis], df[yaxis], marker=".")
    ax.set_xlabel(r"Bp $-$ Rp", fontsize=16)
    if yaxis=="abs_gmag":
        ylabel = r"M$_{\mathrm{G}}$"
        ax.invert_yaxis()
    else:
        ylabel = "Teff [K]"
        raise NotImplementedError

    ax.set_ylabel(ylabel, fontsize=16)

    text = len(df[[xaxis, yaxis]].dropna())
    ax.text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax.transAxes)
    return ax

def plot_hrd_spectral_types():
    """
    """
    df = get_mamajek_table()
    classes = []
    for idx,g in df.assign(SpT2=df['#SpT'].apply(lambda x: x[0])).groupby(by='SpT2'):
        classes.append(idx)
        x = g['logT'].astype(float)
        y = g['logL'].astype(float)
        pl.plot(x, y,label=idx)
    pl.ylabel(r'$\log_{10}$ (L/L$_{\odot}$)')
    pl.xlabel(r'$\log_{10}$ (T$_{\rm{eff}}$/K)')
    pl.legend()
    pl.gca().invert_xaxis()
    return fig

def plot_xyz_3d(
    df,
    target_gaiaid=None,
    match_id=True,
    df_target=None,
    xlim=None,
    ylim=None,
    zlim=None,
    figsize=(10, 10)
):
    """plot 3-d position in galactocentric frame

    Parameters
    ----------
    df : pandas.DataFrame
        contains ra, dec & parallax columns
    target_gaiaid : int
        target gaia DR2 id
    xlim,ylim,zlim : tuple
        lower and upper bounds
    """
    fig = pl.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(30, 120)

    coords = SkyCoord(
        ra=df.ra.values * u.deg,
        dec=df.dec.values * u.deg,
        distance=Distance(parallax=df.parallax.values * u.mas),
    )
    xyz = coords.galactocentric
    df["x"] = xyz.x
    df["y"] = xyz.y
    df["z"] = xyz.z

    idx1 = np.zeros_like(df.x, dtype=bool)
    if xlim:
        assert isinstance(xlim, tuple)
        idx1 = (df.x > xlim[0]) & (df.x < xlim[1])
    idx2 = np.zeros_like(df.y, dtype=bool)
    if ylim:
        assert isinstance(ylim, tuple)
        idx2 = (df.y > ylim[0]) & (df.y < ylim[1])
    idx3 = np.zeros_like(df.z, dtype=bool)
    if zlim:
        assert isinstance(zlim, tuple)
        idx3 = (df.z > zlim[0]) & (df.z < zlim[1])
    idx = idx1 | idx2 | idx3
    ax.scatter(xs=df[idx].x, ys=df[idx].y, zs=df[idx].z, marker=".", alpha=0.5)
    idx = df.source_id == target_gaiaid
    ax.scatter(
        xs=df[idx].x, ys=df[idx].y, zs=df[idx].z, marker=r"$\star$", c="r", s=300
    )
    pl.setp(ax, xlabel="X", ylabel="Y", zlabel="Z")
    return fig


def plot_depth_dmag(gaia_catalog, gaiaid, depth, kmax=1.0, ax=None):
    """
    gaia_catalog : pandas.DataFrame
        gaia catalog
    gaiaid : int
        target gaia DR2 id
    depth : float
        observed transit depth
    kmax : float
        maximum depth
    """
    good, bad, dmags = [], [], []
    idx = gaia_catalog.source_id.isin([gaiaid])
    target_gmag = gaia_catalog.iloc[idx]["phot_g_mean_mag"]
    for id, mag in gaia_catalog[["source_id", "phot_g_mean_mag"]].values:
        if int(id) != gaiaid:
            dmag = mag - target_gmag
            gamma = 1 + 10 ** (0.4 * dmag)
            pl.plot(dmag, kmax / gamma, "b.")
            dmags.append(dmag)
            if depth > kmax / gamma:
                # observed depth is too deep to have originated from the secondary star
                good.append(id)
            else:
                # uncertain signal source
                bad.append(id)
    if ax is None:
        fig, ax = pl.subplots(1, 1)
    ax.axhline(depth, 0, 1, c="k", ls="--")
    dmags = np.linspace(min(dmags), max(dmags), 100)
    gammas = 1 + 10 ** (0.4 * dmags)
    ax.plot(dmags, kmax / gammas, "r-")
    ax.set_yscale("log")
    return ax


def plot_interactive(parallax_cut=2):
    """show altair plots of TOI and clusters

    Parameters
    ----------
    plx_cut : float
        parallax cut in mas; default=2 mas < 100pc
    """
    try:
        import altair as alt
    except ModuleNotFoundError:
        print("pip install altair")
    cc = ClusterCatalog(verbose=False)

    # get Bouma catalog
    df0 = cc.query_catalog(name="Bouma2019", return_members=False)
    idx = df0.parallax >= parallax_cut
    df0 = df0.loc[idx]
    df0["distance"] = Distance(parallax=df0["parallax"].values * u.mas).pc
    # plot Bouma catalog
    chart0 = (
        alt.Chart(df0)
        .mark_point(color="red")
        .encode(
            x=alt.X(
                "ra:Q", axis=alt.Axis(title="RA"), scale=alt.Scale(domain=[0, 360])
            ),
            y=alt.Y(
                "dec:Q", axis=alt.Axis(title="Dec"), scale=alt.Scale(domain=[-90, 90])
            ),
            tooltip=[
                "Cluster:N",
                "distance:Q",
                "parallax:Q",
                "pmra:Q",
                "pmdec:Q",
                "count:Q",
            ],
        )
    )

    # get TOI list
    toi = get_tois(verbose=False, clobber=False)
    toi["TIC_ID"] = toi["TIC ID"]
    toi["RA"] = Angle(toi["RA"].values, unit="hourangle").deg
    toi["Dec"] = Angle(toi["Dec"].values, unit="deg").deg
    # plot TOI
    chart1 = (
        alt.Chart(toi, title="TOI")
        .transform_calculate(
            # FIXME: url below doesn't work in pop-up chart
            url="https://exofop.ipac.caltech.edu/tess/target.php?id="
            + alt.datum.TIC_ID
        )
        .mark_point(color="black")
        .encode(
            x=alt.X(
                "RA:Q", axis=alt.Axis(title="RA"), scale=alt.Scale(domain=[0, 360])
            ),
            y=alt.Y(
                "Dec:Q", axis=alt.Axis(title="Dec"), scale=alt.Scale(domain=[-90, 90])
            ),
            tooltip=[
                "TOI:Q",
                "TIC ID:Q",
                "url:N",
                "Stellar Distance (pc):Q",
                "PM RA (mas/yr):Q",
                "PM Dec (mas/yr):Q",
            ],
        )
        .properties(width=800, height=400)
        .interactive()
    )

    # plot cluster members
    df2 = cc.query_catalog(name="CantatGaudin2018", return_members=True)
    idx = df2.parallax >= parallax_cut
    df2 = df2.loc[idx]
    # skip other members
    df2 = df2.iloc[::10, :]
    chart2 = (
        alt.Chart(df2)
        .mark_circle()
        .encode(
            x="ra:Q",
            y="dec:Q",
            color="Cluster:N",
            tooltip=[
                "source_id:Q",
                "parallax:Q",
                "pmra:Q",
                "pmdec:Q",
                "phot_g_mean_mag:Q",
            ],
        )
    )

    return chart2 + chart1 + chart0

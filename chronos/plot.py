# -*- coding: utf-8 -*-
r"""
classes for plotting cluster properties
"""
import sys
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
import lightkurve as lk
from scipy.ndimage import zoom

# from transitleastsquares import final_T0_fit
from astropy.coordinates import Angle, SkyCoord, Distance
from astropy.visualization import ZScaleInterval
from astroquery.mast import Catalogs
from astropy.wcs import WCS
import astropy.units as u
from astroquery.skyview import SkyView
from astroplan.plots import plot_finder_image
from astropy.timeseries import LombScargle
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import deepdish as dd

# Import from package
from chronos.cluster import ClusterCatalog, Cluster
from chronos.constants import Kepler_pix_scale, TESS_pix_scale
from chronos.utils import (
    get_toi,
    get_tois,
    PadWithZeros,
    get_mamajek_table,
    parse_aperture_mask,
    is_point_inside_mask,
    is_gaiaid_in_cluster,
    get_fluxes_within_mask,
    get_rotation_period,
)

__all__ = [
    "plot_tls",
    "plot_odd_even",
    "plot_hrd_spectral_types",
    "plot_pdc_sap_comparison",
    "plot_rotation_period",
    "plot_possible_NEBs",
    "plot_interactive",
    "plot_aperture_outline",
    "plot_gaia_sources_on_survey",
    "plot_gaia_sources_on_tpf",
]


def plot_cluster_map(
    target_coord=None,
    catalog_name="Bouma2019",
    cluster_name=None,
    offset=10,
    ax=None,
):
    """
    """
    tra = target_coord.ra.deg
    tdec = target_coord.dec.deg
    if ax is None:
        fig, ax = pl.subplot(1, 1, figsize=(5, 5))
    if cluster_name is None:
        cc = ClusterCatalog(catalog_name)
        cat = cc.query_catalog()
        coords = SkyCoord(
            ra=cat["ra"],
            dec=cat["dec"],
            distance=cat["distance"],
            unit=("deg", "deg", "pc"),
        )
        ax.scatter(coords.ra.deg, coords.dec.deg, "ro")
    else:
        c = Cluster(catalog_name=catalog_name, cluster_name=cluster_name)
        mem = c.query_cluster_members()
        rsig = mem["ra"].std()
        dsig = mem["dec"].std()
        r = np.sqrt(rsig ** 2 + dsig ** 2)
        circle = pl.Circle((tra, tdec), r, color="r")
        ax.plot(mem["ra"], mem["dec"], "r.", alpha=0.1)
        ax.add_artist(circle)
    ax.plot(tra, tdec, "bx")
    ax.ylim(tdec - offset, tdec + offset)
    ax.xlim(tra - offset, tra + offset)
    return fig


def plot_orientation_on_tpf(tpf, ax=None):
    """
    Plot the orientation arrows on tpf

    Returns
    -------
    tpf read from lightkurve

    """
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=(5, 5))
    mean_tpf = np.mean(tpf.flux, axis=0)
    zmin, zmax = ZScaleInterval(contrast=0.5)
    ax.matshow(mean_tpf, vmin=zmin, vmax=zmax, origin="lower")
    _ = plot_orientation(tpf, ax=ax)
    return ax


def plot_orientation(tpf, ax):
    """overlay orientation arrows on tpf plot
    """
    nx, ny = tpf.flux.shape[1:]
    x0, y0 = tpf.column + int(0.9 * nx), tpf.row + int(0.2 * nx)
    # East
    tmp = tpf.get_coordinates()
    ra00, dec00 = tmp[0][0][0][0], tmp[1][0][0][0]
    ra10, dec10 = tmp[0][0][0][-1], tmp[1][0][0][-1]
    theta = np.arctan((dec10 - dec00) / (ra10 - ra00))
    if (ra10 - ra00) < 0.0:
        theta += np.pi
    # theta = -22.*np.pi/180.
    x1, y1 = 1.0 * np.cos(theta), 1.0 * np.sin(theta)
    ax.arrow(x0, y0, x1, y1, head_width=0.2, color="white")
    ax.text(x0 + 1.5 * x1, y0 + 1.5 * y1, "E", color="white")
    # North
    theta = theta + 90.0 * np.pi / 180.0
    x1, y1 = 1.0 * np.cos(theta), 1.0 * np.sin(theta)
    ax.arrow(x0, y0, x1, y1, head_width=0.2, color="white")
    ax.text(x0 + 1.5 * x1, y0 + 1.5 * y1, "N", color="white")
    return ax


def plot_gaia_sources_on_tpf(
    tpf,
    target_gaiaid,
    gaia_sources=None,
    sap_mask="pipeline",
    depth=None,
    kmax=1,
    dmag_limit=8,
    fov_rad=None,
    cmap="viridis",
    figsize=None,
    ax=None,
    pix_scale=TESS_pix_scale,
    **mask_kwargs,
):
    """
    plot gaia sources brighter than dmag_limit; only annotated with starids
    are those that are bright enough to cause reproduce the transit depth;
    starids are in increasing separation

    dmag_limit : float
        maximum delta mag to consider; computed based on depth if None
    """
    assert target_gaiaid is not None
    img = np.nanmedian(tpf.flux, axis=0)
    # make aperture mask
    mask = parse_aperture_mask(tpf, sap_mask=sap_mask, **mask_kwargs)
    ax = plot_aperture_outline(
        img, mask=mask, imgwcs=tpf.wcs, figsize=figsize, cmap=cmap, ax=ax
    )
    if fov_rad is None:
        nx, ny = tpf.shape[1:]
        diag = np.sqrt(nx ** 2 + ny ** 2)
        fov_rad = (0.4 * diag * pix_scale).to(u.arcmin)

    if gaia_sources is None:
        target_coord = SkyCoord(
            ra=tpf.header["RA_OBJ"], dec=tpf.header["DEC_OBJ"], unit="deg"
        )
        gaia_sources = Catalogs.query_region(
            target_coord, radius=fov_rad, catalog="Gaia", version=2
        ).to_pandas()
    assert len(gaia_sources) > 1, "gaia_sources contains single entry"
    # find sources within mask
    # target is assumed to be the first row
    idx = gaia_sources["source_id"].astype(int).isin([target_gaiaid])
    target_gmag = gaia_sources.loc[idx, "phot_g_mean_mag"].values[0]
    if depth is not None:
        # compute delta mag limit given transit depth
        dmag_limit = (
            np.log10(kmax / depth - 1) if dmag_limit is None else dmag_limit
        )

        # get min_gmag inside mask
        ra, dec = gaia_sources[["ra", "dec"]].values.T
        pix_coords = tpf.wcs.all_world2pix(np.c_[ra, dec], 0)
        contour_points = measure.find_contours(mask, level=0.1)[0]
        isinside = [
            is_point_inside_mask(contour_points, pix) for pix in pix_coords
        ]
        min_gmag = gaia_sources.loc[isinside, "phot_g_mean_mag"].min()
        if (target_gmag - min_gmag) != 0:
            print(
                f"target Gmag={target_gmag:.2f} is not the brightest within aperture (Gmag={min_gmag:.2f})"
            )
    else:
        min_gmag = gaia_sources.phot_g_mean_mag.min()  # brightest
        dmag_limit = (
            gaia_sources.phot_g_mean_mag.max()
            if dmag_limit is None
            else dmag_limit
        )

    base_ms = 128.0  # base marker size
    starid = 1
    # if very crowded, plot only top N
    gmags = gaia_sources.phot_g_mean_mag
    dmags = gmags - target_gmag
    rank = np.argsort(dmags.values)
    for index, row in gaia_sources.iterrows():
        # FIXME: why some indexes are missing?
        ra, dec, gmag, id = row[["ra", "dec", "phot_g_mean_mag", "source_id"]]
        dmag = gmag - target_gmag
        pix = tpf.wcs.all_world2pix(np.c_[ra, dec], 0)[0]
        contour_points = measure.find_contours(mask, level=0.1)[0]

        color, alpha = "red", 1.0
        # change marker color and transparency depending on the location and dmag
        if is_point_inside_mask(contour_points, pix):
            if int(id) == int(target_gaiaid):
                # plot x on target
                ax.plot(
                    pix[1],
                    pix[0],
                    marker="x",
                    ms=base_ms / 16,
                    c="k",
                    zorder=3,
                )
            if depth is not None:
                # compute flux ratio with respect to brightest star
                gamma = 1 + 10 ** (0.4 * (min_gmag - gmag))
                if depth > kmax / gamma:
                    # orange if flux is insignificant
                    color = "C1"
        else:
            # outside aperture
            color, alpha = "C1", 0.5

        ax.scatter(
            pix[1],
            pix[0],
            s=base_ms / 2 ** dmag,  # fainter -> smaller
            c=color,
            alpha=alpha,
            zorder=2,
            edgecolor=None,
        )
        # choose which star to annotate
        if len(gmags) < 20:
            # sparse: annotate all
            ax.text(pix[1], pix[0], str(starid), color="white", zorder=100)
        elif len(gmags) > 50:
            # crowded: annotate only 15 smallest dmag ones
            if rank[starid - 1] < 15:
                ax.text(pix[1], pix[0], str(starid), color="white", zorder=100)
            elif (color == "red") & (dmag < dmag_limit):
                # plot if within aperture and significant source of dilution
                ax.text(pix[1], pix[0], str(starid), color="white", zorder=100)
        elif color == "red":
            # neither sparse nor crowded
            # annotate if inside aperture
            ax.text(pix[1], pix[0], str(starid), color="white", zorder=100)
        starid += 1
    # Make legend with 4 sizes representative of delta mags
    dmags = dmags[dmags < dmag_limit]
    _, dmags = pd.cut(dmags, 3, retbins=True)
    for dmag in dmags:
        size = base_ms / 2 ** dmag
        # -1, -1 is outside the fov
        # dmag = 0 if float(dmag)==0 else 0
        ax.scatter(
            -1,
            -1,
            s=size,
            c="red",
            alpha=0.6,
            edgecolor=None,
            zorder=10,
            clip_on=True,
            label=r"$\Delta m= $" + f"{dmag:.1f}",
        )
    ax.legend(fancybox=True, framealpha=0.5)
    # set img limits
    xdeg = (nx * pix_scale).to(u.arcmin)
    ydeg = (ny * pix_scale).to(u.arcmin)
    pl.setp(
        ax, xlim=(0, nx), ylim=(0, ny), xlabel=f"({xdeg:.2f} x {ydeg:.2f})"
    )
    return ax


def plot_gaia_sources_on_survey(
    tpf,
    target_gaiaid,
    gaia_sources=None,
    fov_rad=None,
    depth=0.0,
    kmax=1.0,
    sap_mask="pipeline",
    survey="DSS2 Red",
    verbose=True,
    ax=None,
    figsize=None,
    pix_scale=TESS_pix_scale,
    **mask_kwargs,
):
    """Plot (superpose) Gaia sources on archival image

    Parameters
    ----------
    target_coord : astropy.coordinates
        target coordinate
    gaia_sources : pd.DataFrame
        gaia sources table
    fov_rad : astropy.unit
        FOV radius
    survey : str
        image survey; see from astroquery.skyview import SkyView;
        SkyView.list_surveys()
    verbose : bool
        print texts
    ax : axis
        subplot axis
    kwargs : dict
        keyword arguments for aper_radius, percentile
    Returns
    -------
    ax : axis
        subplot axis
    """
    assert target_gaiaid is not None
    ny, nx = tpf.flux.shape[1:]
    if fov_rad is None:
        diag = np.sqrt(nx ** 2 + ny ** 2)
        fov_rad = (0.4 * diag * pix_scale).to(u.arcmin)
    target_coord = SkyCoord(ra=tpf.ra * u.deg, dec=tpf.dec * u.deg)
    if gaia_sources is None:
        gaia_sources = Catalogs.query_region(
            target_coord, radius=fov_rad, catalog="Gaia", version=2
        ).to_pandas()
    assert len(gaia_sources) > 1, "gaia_sources contains single entry"
    # make aperture mask
    mask = parse_aperture_mask(tpf, sap_mask=sap_mask, **mask_kwargs)
    maskhdr = tpf.hdu[2].header
    # make aperture mask outline
    contour = np.zeros((ny, nx))
    contour[np.where(mask)] = 1
    contour = np.lib.pad(contour, 1, PadWithZeros)
    highres = zoom(contour, 100, order=0, mode="nearest")
    extent = np.array([-1, nx, -1, ny])

    if verbose:
        print(
            f"Querying {survey} ({fov_rad:.2f} x {fov_rad:.2f}) archival image"
        )
    # -----------create figure---------------#
    if ax is None:
        # get img hdu for subplot projection
        hdu = SkyView.get_images(
            position=target_coord.icrs,
            coordinates="icrs",
            survey=survey,
            radius=fov_rad,
            grid=False,
        )[0][0]
        fig = pl.figure(figsize=figsize)
        # define scaling in projection
        ax = fig.add_subplot(111, projection=WCS(hdu.header))
    # plot survey img
    nax, hdu = plot_finder_image(
        target_coord, ax=ax, fov_radius=fov_rad, survey=survey, reticle=False
    )
    imgwcs = WCS(hdu.header)
    mx, my = hdu.data.shape
    # plot mask
    _ = ax.contour(
        highres,
        levels=[0.5],
        extent=extent,
        origin="lower",
        colors="C0",
        transform=ax.get_transform(WCS(maskhdr)),
    )
    idx = gaia_sources["source_id"].astype(int).isin([target_gaiaid])
    target_gmag = gaia_sources.loc[idx, "phot_g_mean_mag"].values[0]

    for index, row in gaia_sources.iterrows():
        marker, s = "o", 100
        r, d, mag, id = row[["ra", "dec", "phot_g_mean_mag", "source_id"]]
        pix = imgwcs.all_world2pix(np.c_[r, d], 1)[0]
        if int(id) != int(target_gaiaid):
            gamma = 1 + 10 ** (0.4 * (mag - target_gmag))
            if depth > kmax / gamma:
                # too deep to have originated from secondary star
                edgecolor = "C1"
                alpha = 1  # 0.5
            else:
                # possible NEBs
                edgecolor = "C3"
                alpha = 1
        else:
            s = 200
            edgecolor = "C2"
            marker = "s"
            alpha = 1
        nax.scatter(
            pix[0],
            pix[1],
            marker=marker,
            s=s,
            edgecolor=edgecolor,
            alpha=alpha,
            facecolor="none",
        )
    # set img limits
    pl.setp(
        nax,
        xlim=(0, mx),
        ylim=(0, my),
        title="{0} ({1:.2f}' x {1:.2f}')".format(survey, fov_rad.value),
    )
    return ax


def plot_aperture_outline(
    img, mask, ax=None, imgwcs=None, cmap="viridis", figsize=None
):
    """
    see https://github.com/rodluger/everest/blob/56f61a36625c0d9a39cc52e96e38d257ee69dcd5/everest/standalone.py
    """
    interval = ZScaleInterval(contrast=0.5)
    ny, nx = mask.shape
    contour = np.zeros((ny, nx))
    contour[np.where(mask)] = 1
    #     contour = np.lib.pad(contour, 1, PadWithZeros)
    highres = zoom(contour, 100, order=0, mode="nearest")
    extent = np.array([-1, nx, -1, ny])

    if ax is None:
        fig, ax = pl.subplots(
            subplot_kw={"projection": imgwcs}, figsize=figsize
        )
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
    _ = ax.contour(
        highres,
        # levels=[0.5],
        linewidths=3,
        extent=extent,
        origin="lower",
        colors="C6",
    )
    zmin, zmax = interval.get_limits(img)
    ax.matshow(
        img, origin="lower", cmap=cmap, vmin=zmin, vmax=zmax, extent=extent
    )
    # verts = cs.allsegs[0][0]
    return ax


def plot_possible_NEBs(gaia_sources, depth, gaiaid=None, kmax=1.0, ax=None):
    """
    """
    assert len(gaia_sources) > 1, "gaia_sources contains single entry"
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=(5, 5))

    if gaiaid is None:
        # nearest match (first entry row=0) is assumed as the target
        gaiaid = gaia_sources.iloc[0]["source_id"]
    idx = gaia_sources.source_id.isin([gaiaid])
    target_gmag = gaia_sources.loc[idx, "phot_g_mean_mag"].values[0]

    good, bad, dmags = [], [], []
    for index, row in gaia_sources.iterrows():
        id, mag = row[["source_id", "phot_g_mean_mag"]]
        if int(id) != gaiaid:
            dmag = mag - target_gmag
            gamma = 1 + 10 ** (0.4 * dmag)
            ax.plot(dmag, kmax / gamma, "b.")
            dmags.append(dmag)
            if depth > kmax / gamma:
                # observed depth is too deep to have originated from the secondary star
                good.append(id)
            else:
                # uncertain signal source
                bad.append(id)
    ax.axhline(depth, 0, 1, c="k", ls="--")
    dmags = np.linspace(min(dmags), max(dmags), 100)
    gammas = 1 + 10 ** (0.4 * dmags)

    nbad = len(bad)
    ax.plot(dmags, kmax / gammas, "r-", label=f"potential NEBs={nbad}")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta$Gmag")
    ax.set_ylabel("Eclipse depth")
    ax.legend()
    return ax


def plot_rotation_period(
    time,
    flux,
    err=None,
    mask=None,
    method="lombscargle",
    min_per=0.5,
    max_per=30,
    npoints=20,
    xlims=None,
    ylims=None,
    figsize=(10, 5),
    title=None,
):
    """
    method : str
        lombscargle or acf (autocorrelation function)
    """
    fig, ax = pl.subplots(1, 2, figsize=figsize, constrained_layout=True)
    if mask is not None:
        time, flux = time[~mask], flux[~mask]
        err = None if err is None else err[~mask]
    if method == "lombscargle":
        ls = LombScargle(time, flux, dy=err)
        frequencies, powers = ls.autopower(
            minimum_frequency=1.0 / max_per, maximum_frequency=1.0 / min_per
        )
        best_freq = frequencies[np.argmax(powers)]
        peak_period = 1.0 / best_freq
        periods = 1.0 / frequencies
    elif method == "acf":
        raise NotImplementedError("Method not yet available")
    else:
        raise ValueError("Use method='lombscargle'")
    # fit a gaussian to lombscargle power
    prot, prot_err = get_rotation_period(
        time,
        flux,
        min_per=min_per,
        max_per=max_per,
        npoints=npoints,
        plot=False,
    )

    # left: periodogram
    n = 0
    ax[n].plot(periods, powers, "k-")
    ax[n].axvline(
        peak_period, 0, 1, ls="--", c="r", label=f"peak={peak_period:.2f}"
    )
    ax[n].axvline(
        prot, 0, 1, ls="-", c="r", label=f"fit={prot:.2f}+/-{prot_err:.2f}"
    )
    ax[n].legend(title="Best period [d]")
    ax[n].set_xscale("log")
    ax[n].set_xlabel("Period [days]")
    ax[n].set_ylabel("Lomb-Scargle Power")

    # right: phase-folded lc and sinusoidal model
    n = 1
    offset = 0.5
    t_fit = np.linspace(0, 1, 100) - offset
    y_fit = ls.model(t_fit * peak_period - peak_period / 2, best_freq)
    ax[n].plot(
        t_fit * peak_period, y_fit, "r-", lw=3, label="sine model", zorder=3
    )
    # fold data
    phase = ((time / peak_period) % 1) - offset

    a = ax[n].scatter(
        phase * peak_period, flux, c=time, cmap=pl.get_cmap("Blues")
    )
    pl.colorbar(a, ax=ax[n], label=f"Time [BTJD]")
    ax[n].legend()
    if xlims is None:
        ax[n].set_xlim(-peak_period / 2, peak_period / 2)
    else:
        ax[n].set_xlim(*xlims)
    if ylims is not None:
        ax[n].set_ylim(*ylims)
    ax[n].set_ylabel("Normalized Flux")
    ax[n].set_xlabel("Phase [days]")
    fig.suptitle(title)
    return fig


def plot_tls(results, period=None, plabel=None, figsize=None):
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
    fig, ax = pl.subplots(2, 1, figsize=figsize)

    n = 0
    label = f"TLS={results.period:.3}"
    ax[n].axvline(results.period, alpha=0.4, lw=3, label=label)
    ax[n].set_xlim(np.min(results.periods), np.max(results.periods))

    for i in range(2, 10):
        ax[n].axvline(i * results.period, alpha=0.4, lw=1, linestyle="dashed")
        ax[n].axvline(results.period / i, alpha=0.4, lw=1, linestyle="dashed")
    ax[n].set_ylabel(r"SDE")
    ax[n].set_xlabel("Period (days)")
    ax[n].plot(results.periods, results.power, color="black", lw=0.5)
    ax[n].set_xlim(0, max(results.periods))
    if period is not None:
        ax[n].axvline(period, 0, 1, ls="--", c="r", label=plabel)
    ax[n].legend(title="best period (d)")

    n = 1
    ax[n].plot(
        results.model_folded_phase - 0.5, results.model_folded_model, color="b"
    )
    ax[n].scatter(
        results.folded_phase - 0.5,
        results.folded_y,
        color="k",
        s=10,
        alpha=0.5,
        zorder=2,
    )
    xlim = 3 * results.duration / results.period
    ax[n].set_xlim(-xlim, xlim)
    ax[n].set_xlabel("Phase")
    ax[n].set_ylabel("Relative flux")
    fig.tight_layout()
    return fig


def plot_odd_even(flat, period, epoch, yline=None, figsize=(8, 4)):
    """
    """
    fig, axs = pl.subplots(
        1, 2, figsize=figsize, sharey=True, constrained_layout=True
    )
    fold = flat.fold(period=period, t0=epoch)

    ax = axs[0]
    fold[fold.even_mask].scatter(label="even", ax=ax)
    if yline is not None:
        ax.axhline(yline, 0, 1, lw=2, ls="--", c="k")

    ax = axs[1]
    fold[fold.odd_mask].scatter(label="odd", ax=ax)
    if yline is not None:
        ax.axhline(yline, 0, 1, lw=2, ls="--", c="k")
    ax.set_ylabel("")
    fig.subplots_adjust(wspace=0)
    return fig


def plot_pdc_sap_comparison(toiid, sector=None):
    toi = get_toi(toi=toiid, verbose=False)
    period = toi["Period (days)"].values[0]
    t0 = toi["Epoch (BJD)"].values[0]
    tic = toi["TIC ID"].values[0]

    lcf = lk.search_lightcurvefile(
        f"TIC {tic}", sector=sector, mission="TESS"
    ).download()
    if lcf is not None:
        sap = lcf.SAP_FLUX.remove_nans().normalize()
        pdcsap = lcf.PDCSAP_FLUX.remove_nans().normalize()

        ax = sap.bin(11).fold(period=period, t0=t0).scatter(label="SAP")
        _ = (
            pdcsap.bin(11)
            .fold(period=period, t0=t0)
            .scatter(ax=ax, label="PDCSAP")
        )
        # ax.set_xlim(-0.1,0.1)
        ax.set_title(f"TOI {toiid} (sector {sap.sector})")
    return lcf, ax


def plot_hrd_spectral_types(
    x=None,
    y=None,
    c=None,
    cmap="viridis",
    invert_xaxis=True,
    invert_yaxis=False,
    **plot_kwargs,
):
    """
    """
    df = get_mamajek_table()
    fig, ax = pl.subplots(1, 1, **plot_kwargs)

    if (x is not None) & (y is not None):
        _ = df.plot.scatter(x=x, y=y, c=c, ax=ax, cmap=cmap)
        # _ = df.plot.scatter(x='V-Ks', y='M_Ks', c='R_Rsun', cmap='viridis')
        # ax.axhline(6.7, 0, 1, ls='--', c='k')
        # ax.annotate(s='fully convective', xy=(7, 8), fontsize=12)
        if invert_xaxis:
            ax.invert_xaxis()
        if invert_yaxis:
            ax.invert_yaxis()
        ax.set_ylabel(y)
        ax.set_xlabel(x)
    else:
        classes = []
        for idx, g in df.assign(SpT2=df["#SpT"].apply(lambda x: x[0])).groupby(
            by="SpT2"
        ):
            classes.append(idx)
            x = g["logT"].astype(float)
            y = g["logL"].astype(float)
            ax.plot(x, y, label=idx)
        ax.set_ylabel(r"$\log_{10}$ (L/L$_{\odot}$)")
        ax.set_xlabel(r"$\log_{10}$ (T$_{\rm{eff}}$/K)")
        ax.legend()
        ax.invert_xaxis()
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
    for index, row in gaia_catalog.iterrows():
        id, mag = row[["source_id", "phot_g_mean_mag"]]
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
    ax.axhline(depth, 0, 1, c="k", ls="--", label="TESS depth")
    dmags = np.linspace(min(dmags), max(dmags), 100)
    gammas = 1 + 10 ** (0.4 * dmags)
    ax.plot(dmags, kmax / gammas, "r-")
    ax.set_yscale("log")
    return ax


def plot_interactive(
    catalog_name="CantatGaudin2020",
    min_parallax=1.5,
    thin=10,
    width=800,
    height=400,
):
    """show altair plots of TOI and clusters

    Parameters
    ----------
    plx_cut : float
        parallax cut in mas; default=2 mas < 100pc
    thin : integer
        thinning factor to use ony every nth cluster member
    """
    try:
        import altair as alt
    except ModuleNotFoundError:
        print("pip install altair")

    if sys.argv[-1].endswith("json"):
        print("import altair; altair.notebook()")

    cc = ClusterCatalog(verbose=False)
    df0 = cc.query_catalog(catalog_name=catalog_name, return_members=False)
    df2 = cc.query_catalog(catalog_name=catalog_name, return_members=True)
    # add members count from df2 in df0
    # counts = df2.groupby('Cluster').size()
    # counts.name = 'Nstars'
    # counts = counts.reset_index()
    # df0 = pd.merge(df0, counts, how='outer')
    idx = df0.parallax >= min_parallax
    df0 = df0.loc[idx]
    df0["distance"] = Distance(parallax=df0["parallax"].values * u.mas).pc
    # plot catalog
    chart0 = (
        alt.Chart(df0)
        .mark_point(color="red")
        .encode(
            x=alt.X(
                "ra:Q",
                axis=alt.Axis(title="RA"),
                scale=alt.Scale(domain=[0, 360]),
            ),
            y=alt.Y(
                "dec:Q",
                axis=alt.Axis(title="Dec"),
                scale=alt.Scale(domain=[-90, 90]),
            ),
            tooltip=[
                "Cluster:N",
                "distance:Q",
                "parallax:Q",
                "pmra:Q",
                "pmdec:Q",
                "Nstars:Q",
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
                "RA:Q",
                axis=alt.Axis(title="RA"),
                scale=alt.Scale(domain=[0, 360]),
            ),
            y=alt.Y(
                "Dec:Q",
                axis=alt.Axis(title="Dec"),
                scale=alt.Scale(domain=[-90, 90]),
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
        .properties(width=width, height=height)
        .interactive()
    )

    # plot cluster members
    idx = df2.parallax >= min_parallax
    df2 = df2.loc[idx]
    # skip other members
    df2 = df2.iloc[::thin, :]
    chart2 = (
        alt.Chart(df2)
        .mark_circle()
        .encode(
            x="ra:Q",
            y="dec:Q",
            color="Cluster:N",
            tooltip=[
                "source_id:O",
                "parallax:Q",
                "pmra:Q",
                "pmdec:Q",
                "phot_g_mean_mag:Q",
            ],
        )
    )

    return chart2 + chart1 + chart0

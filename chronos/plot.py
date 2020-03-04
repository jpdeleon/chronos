# -*- coding: utf-8 -*-

r"""
classes for plotting cluster properties
"""
# Import standard library
import sys
import os
import logging
import itertools
import traceback

# Import modules
# from matplotlib.figure import Figure
# from matplotlib.image import AxesImage
# from loguru import logger
import numpy as np
import matplotlib.pyplot as pl
import lightkurve as lk
from scipy.ndimage import zoom
from transitleastsquares import transitleastsquares as tls
from astropy.coordinates import Angle, SkyCoord, Distance
from astropy.visualization import ZScaleInterval
from astroquery.mast import Catalogs
from astropy.wcs import WCS
import astropy.units as u
from astroplan.plots import plot_finder_image
from astropy.timeseries import LombScargle
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import deepdish as dd

# Import from package
from chronos.cluster import ClusterCatalog
from chronos.lightcurve import ShortCadence, LongCadence
from chronos.utils import (
    get_transformed_coord,
    get_toi,
    get_tois,
    get_mamajek_table,
    get_absolute_gmag,
    get_absolute_color_index,
    parse_aperture_mask,
    is_point_inside_mask,
    compute_fluxes_within_mask,
)

TESS_pix_scale = 21 * u.arcsec  # /pix

log = logging.getLogger(__name__)

__all__ = [
    "plot_rdp_pmrv",
    "plot_xyz_uvw",
    "plot_cmd",
    "plot_hrd",
    "plot_tls",
    "plot_odd_even",
    "plot_hrd_spectral_types",
    "plot_pdc_sap_comparison",
    "plot_lomb_scargle",
    "plot_possible_NEBs",
    "plot_interactive",
    "plot_aperture_outline",
    "plot_gaia_sources_on_survey",
    "plot_gaia_sources_on_tpf",
    "make_tql",
]


def make_tql(
    gaiaid=None,
    toiid=None,
    ticid=None,
    name=None,
    sector=None,
    cadence="long",
    sap_mask=None,
    aper_radius=1,
    threshold_sigma=5,
    percentile=90,
    cutout_size=(15, 15),
    quality_bitmask="default",
    apply_data_quality_mask=True,
    window_length=31,
    savefig=False,
    savetls=False,
    outdir=".",
    verbose=False,
    clobber=False,
):
    try:
        if cadence == "long":
            if sap_mask is None:
                sap_mask = "square"
            l = LongCadence(
                gaiaDR2id=gaiaid,
                toiid=toiid,
                ticid=ticid,
                name=name,
                sector=sector,
                sap_mask=sap_mask,
                aper_radius=aper_radius,
                threshold_sigma=threshold_sigma,
                percentile=percentile,
                cutout_size=cutout_size,
                quality_bitmask=quality_bitmask,
                apply_data_quality_mask=apply_data_quality_mask,
                verbose=verbose,
                clobber=clobber,
            )
            if verbose:
                print("Querying Tesscut\n")
            tpf = l.get_tpf_tesscut(sector=l.sector)
        elif cadence == "short":
            if sap_mask is None:
                sap_mask = "pipeline"
            l = ShortCadence(
                gaiaDR2id=gaiaid,
                toiid=toiid,
                ticid=ticid,
                name=name,
                sector=sector,
                sap_mask=sap_mask,
                aper_radius=aper_radius,
                threshold_sigma=threshold_sigma,
                percentile=percentile,
                quality_bitmask=quality_bitmask,
                apply_data_quality_mask=apply_data_quality_mask,
                verbose=verbose,
                clobber=clobber,
            )
            if verbose:
                print("Querying Tesscut\n")
            tpf, tpf_info = l.get_tpf(sector=l.sector)
        else:
            raise ValueError("Use cadence=(long, short).")

        if (outdir is not None) & (not os.path.exists(outdir)):
            os.makedirs(outdir)

        fig, axs = pl.subplots(2, 3, figsize=(12, 8))
        axs = axs.flatten()

        # +++++++++++++++++++++ax0: tpf
        ax = axs[0]
        if gaiaid is None:
            _ = l.query_gaia_dr2_catalog(return_nearest_xmatch=True)
        if l.gaia_sources is None:
            _ = l.query_gaia_dr2_catalog(radius=120)
        _ = plot_gaia_sources_on_tpf(
            tpf=tpf,
            target_gaiaid=l.gaiaid,
            gaia_sources=l.gaia_sources,
            sap_mask=l.sap_mask,
            aper_radius=l.aper_radius,
            threshold_sigma=l.threshold_sigma,
            ax=ax,
        )
        #     #+++++++++++++++++++++ax1: raw lc
        #     ax = axs[1]
        aper_mask = parse_aperture_mask(
            tpf,
            sap_mask=l.sap_mask,
            aper_radius=l.aper_radius,
            # threshold_sigma=l.threshold_sigma
        )
        raw_lc = tpf.to_lightcurve(aperture_mask=aper_mask)
        #     raw_lc.scatter(ax=ax, label='raw')
        #     ax.legend()

        # +++++++++++++++++++++ax1: bkg-subtracted lc
        if verbose:
            print("Performing background subtraction\n")
        idx = (
            np.isnan(raw_lc.time)
            | np.isnan(raw_lc.flux)
            | np.isnan(raw_lc.flux_err)
        )
        raw_lc = raw_lc[~idx]
        # Make a design matrix and pass it to a linear regression corrector
        regressors = tpf.flux[~idx][:, ~aper_mask]
        dm = (
            lk.DesignMatrix(regressors, name="pixels")
            .pca(nterms=5)
            .append_constant()
        )

        # Regression Corrector Object
        rc = lk.RegressionCorrector(raw_lc)
        bkg_sub_lc = rc.correct(dm)
        #     bkg_sub_lc.normalize().scatter(ax=ax, label='bkg_sub')
        #     bkg_sub_lc.normalize().bin(10).scatter(ax=ax, marker='o', label='bkg_sub (bin=10)')

        # +++++++++++++++++++++ax2: Detrending/ Flattening
        ax = axs[1]
        lc = bkg_sub_lc.normalize().remove_nans().remove_outliers()
        flat, trend = lc.flatten(
            window_length=window_length, return_trend=True
        )
        _ = lc.scatter(ax=ax, label="bkg_sub")
        trend.plot(ax=ax, label="trend", lw=3, c="r")

        ax = axs[2]
        flat.scatter(ax=ax, label="flat")
        flat.bin(10).scatter(ax=ax, label="flat (bin=10)")

        # +++++++++++++++++++++ax3: TLS periodogram
        ax = axs[3]
        lc = flat
        tls_results = tls(lc.time, lc.flux).power()

        label = f"Best period={tls_results.period:.3}"
        ax.axvline(tls_results.period, alpha=0.4, lw=3, label=label)
        ax.set_xlim(np.min(tls_results.periods), np.max(tls_results.periods))

        for i in range(2, 10):
            ax.axvline(
                i * tls_results.period, alpha=0.4, lw=1, linestyle="dashed"
            )
            ax.axvline(
                tls_results.period / i, alpha=0.4, lw=1, linestyle="dashed"
            )
        ax.set_ylabel(r"SDE")
        ax.set_xlabel("Period (days)")
        ax.plot(tls_results.periods, tls_results.power, color="black", lw=0.5)
        ax.set_xlim(0, max(tls_results.periods))
        ax.set_title("TLS Periodogram")
        ax.legend()

        # +++++++++++++++++++++ax4: phase-folded
        ax = axs[4]

        ax.plot(
            tls_results.model_folded_phase,
            tls_results.model_folded_model,
            color="red",
        )
        ax.scatter(
            tls_results.folded_phase,
            tls_results.folded_y,
            color="blue",
            s=10,
            alpha=0.5,
            zorder=2,
        )
        ax.set_xlabel("Phase")
        ax.set_ylabel("Relative flux")
        ax.set_xlim(0.2, 0.8)

        # +++++++++++++++++++++summary
        ax = axs[5]
        tic_params = l.query_tic_catalog(return_nearest_xmatch=True)
        Rp = tls_results["rp_rs"] * tic_params["rad"] * u.Rsun.to(u.Rearth)

        msg = "Candidate Properties\n"
        msg += "-" * 30 + "\n"
        msg += f"SDE={tls_results.SDE:.2f}\n"
        msg += (
            f"Period={tls_results.period:.2f}+/-{tls_results.period_uncertainty:.2f} d"
            + " " * 5
        )
        msg += f"T0={tls_results.T0:.2f} BTJD\n"
        msg += f"Duration={tls_results.duration:.2f} d\n"
        msg += f"Depth={1-tls_results.depth:.4f}\t"
        msg += f"Rp={Rp:.2f} " + r"R$_{\oplus}$" + "\n"
        msg += f"Odd-Even mismatch={tls_results.odd_even_mismatch:.2f}\n"

        msg += "\n" * 2
        msg += "Stellar Properties\n"
        msg += "-" * 30 + "\n"
        msg += f"TIC ID={int(tic_params['ID'])}" + " " * 5
        msg += f"Tmag={tic_params['Tmag']:.2f}\n"
        msg += (
            f"Rstar={tic_params['rad']:.2f}+/-{tic_params['e_rad']:.2f} "
            + r"R$_{\odot}$"
            + " " * 5
        )
        msg += (
            f"Mstar={tic_params['mass']:.2f}+/-{tic_params['e_mass']:.2f} "
            + r"M$_{\odot}$"
            + "\n"
        )
        msg += (
            f"Teff={int(tic_params['Teff'])}+/-{int(tic_params['e_Teff'])} K"
            + " " * 5
        )
        msg += (
            f"logg={tic_params['logg']:.2f}+/-{tic_params['e_logg']:.2f} dex\n"
        )
        msg += (
            r"$\rho$"
            + f"star={tic_params['rho']:.2f}+/-{tic_params['e_rho']:.2f} gcc\n"
        )
        msg += f"Contamination ratio={tic_params['contratio']:.2f}\n"
        ax.text(0, 0, msg, fontsize=10)
        ax.axis("off")

        ticid = tic_params["ID"]
        fig.suptitle(f"TIC {ticid} (sector {l.sector})")
        fig.tight_layout()

        msg = ""
        if savefig:
            fp = os.path.join(outdir, f"tic{ticid}_s{l.sector}_{cadence[0]}c")
            fig.savefig(fp + ".png", bbox_inches="tight")
            msg += f"Saved: {fp}.png\n"
        if savetls:
            tls_results["gaiaid"] = l.gaiaid
            tls_results["ticid"] = l.ticid
            dd.io.save(fp + "_tls.h5", tls_results)
            msg += f"Saved: {fp}_tls.h5"
        if verbose:
            print(msg)
        return fig

    except Exception:
        # Get current system exception
        ex_type, ex_value, ex_traceback = sys.exc_info()
        # Extract unformatter stack traces as tuples
        trace_back = traceback.extract_tb(ex_traceback)

        print(f"Exception type: {ex_type.__name__}")
        print(f"Exception message: {ex_value}")
        # Format stacktrace
        for trace in trace_back:
            print(f"File : {trace[0]}")
            print(f"Line. : {trace[1]}")
            print(f"Func : {trace[2]}")
            print(f"Message : {trace[3]}")


def plot_gaia_sources_on_tpf(
    tpf,
    target_gaiaid,
    gaia_sources=None,
    sap_mask="pipeline",
    depth=None,
    kmax=1,
    ax=None,
    fov_rad=None,
    figsize=None,
    **mask_kwargs,
):
    """
    sources within aperture are red or orange if depth>kmax/gamma; green if outside
    """
    assert target_gaiaid is not None
    img = np.nanmedian(tpf.flux, axis=0)
    # make aperture mask
    mask = parse_aperture_mask(tpf, sap_mask=sap_mask, **mask_kwargs)
    ax = plot_aperture_outline(
        img, mask=mask, imgwcs=tpf.wcs, figsize=figsize, ax=ax
    )
    if fov_rad is None:
        nx, ny = tpf.shape[:2]
        diag = np.sqrt(nx ** 2 + ny ** 2)
        fov_rad = (0.4 * diag * TESS_pix_scale).to(u.arcmin)

    if gaia_sources is None:
        target_coord = SkyCoord(
            ra=tpf.header["RA_OBJ"], dec=tpf.header["DEC_OBJ"], unit="deg"
        )
        gaia_sources = Catalogs.query_region(
            target_coord, radius=fov_rad, catalog="Gaia", version=2
        ).to_pandas()
    assert len(gaia_sources) > 1, "gaia_sources contains single entry"
    # find sources within mask
    if depth is not None:
        # target is assumed to be the first row
        idx = gaia_sources["source_id"].astype(int).isin([target_gaiaid])
        target_gmag = gaia_sources.loc[idx, "phot_g_mean_mag"].values[0]

        # get sources inside mask
        ra, dec = gaia_sources[["ra", "dec"]].values.T
        pix_coords = tpf.wcs.all_world2pix(np.c_[ra, dec], 0)
        contour_points = measure.find_contours(mask, level=0.1)[0]
        isinside = [
            is_point_inside_mask(contour_points, pix) for pix in pix_coords
        ]
        min_gmag = gaia_sources.loc[isinside, "phot_g_mean_mag"].min()
        if (target_gmag - min_gmag) != 0:
            print(
                f"target Gmag={target_gmag:.2f} is not the brightest within aperture (Gmag={min_gmag:.2f  })"
            )

    for index, row in gaia_sources.iterrows():
        ra, dec, gmag, id = row[["ra", "dec", "phot_g_mean_mag", "source_id"]]
        pix = tpf.wcs.all_world2pix(np.c_[ra, dec], 0)[0]
        contour_points = measure.find_contours(mask, level=0.1)[0]

        alpha, marker = 1.0, "o"
        if is_point_inside_mask(contour_points, pix):
            edgecolor = "C3"
            if int(id) == int(target_gaiaid):
                marker = "s"
                edgecolor = "w"
                # ax.plot(pix[1],pix[0], marker='x', ms=20, lw=10, c='w')
            if depth is not None:
                gamma = 1 + 10 ** (0.4 * (min_gmag - gmag))
                if depth > kmax / gamma:
                    edgecolor = "C1"
        else:
            # alpha=0.5
            edgecolor = "C2"
        ax.scatter(
            pix[1],
            pix[0],
            marker=marker,
            s=50,
            edgecolor=edgecolor,
            alpha=alpha,
            facecolor="none",
        )
    # set img limits
    # pl.setp(
    #     ax,
    #     xlim=(min(pix_coords[0]), max(pix_coords[0])),
    #     ylim=(min(pix_coords[1]), max(pix_coords[1])),
    # )
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
        fov_rad = (0.4 * diag * TESS_pix_scale).to(u.arcmin)
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
    #     contour = np.lib.pad(contour, 1, PadWithZeros)
    highres = zoom(contour, 100, order=0, mode="nearest")
    extent = np.array([-1, nx, -1, ny])

    if verbose:
        print(
            f"Querying {survey} ({fov_rad:.2f} x {fov_rad:.2f}) archival image"
        )
    # get img hdu
    nax, hdu = plot_finder_image(
        target_coord, fov_radius=fov_rad, survey=survey, reticle=True
    )
    pl.close()

    # -----------create figure---------------#
    if ax is None:
        fig = pl.figure(figsize=figsize)
        # define scaling in projection
        ax = fig.add_subplot(111, projection=WCS(hdu.header))
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
        transform=nax.get_transform(WCS(maskhdr)),
    )
    idx = gaia_sources["source_id"].astype(int).isin([target_gaiaid])
    target_gmag = gaia_sources.loc[idx, "phot_g_mean_mag"].values[0]

    for index, rows in gaia_sources.iterrows():
        marker, s = "o", 100
        r, d, mag, id = rows[["ra", "dec", "phot_g_mean_mag", "source_id"]]
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


def plot_aperture_outline(img, mask, ax=None, imgwcs=None, figsize=None):
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
        highres, levels=[0.5], extent=extent, origin="lower", colors="C6"
    )
    zmin, zmax = interval.get_limits(img)
    ax.matshow(img, origin="lower", vmin=zmin, vmax=zmax, extent=extent)
    # verts = cs.allsegs[0][0]
    return ax


def plot_possible_NEBs(gaia_params, depth, gaiaid=None, kmax=1.0, ax=None):
    """
    """
    assert len(gaia_params) > 1, "gaia_params contains single entry"
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=(5, 5))

    if gaiaid is None:
        # nearest match (first entry row=0) is assumed as the target
        gaiaid = gaia_params.iloc[0]["source_id"]
    idx = gaia_params.source_id.isin([gaiaid])
    target_gmag = gaia_params.loc[idx, "phot_g_mean_mag"].values[0]

    good, bad, dmags = [], [], []
    for id, mag in gaia_params[["source_id", "phot_g_mean_mag"]].values:
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


def plot_lomb_scargle(
    t,
    f,
    min_per=0.5,
    max_per=30,
    xlims=None,
    ylims=None,
    figsize=(10, 5),
    title=None,
):
    fig, ax = pl.subplots(1, 2, figsize=figsize, constrained_layout=True)
    ls = LombScargle(t, f)
    frequencies, powers = ls.autopower(
        minimum_frequency=1.0 / max_per, maximum_frequency=1.0 / min_per
    )
    best_freq = frequencies[np.argmax(powers)]
    best_period = 1.0 / best_freq
    periods = 1.0 / frequencies
    n = 0
    ax[n].plot(periods, powers, "k-")
    ax[n].axvline(
        best_period, 0, 1, ls="--", c="r", label=f"P_ls={best_period:.2f}"
    )
    ax[n].legend()
    ax[n].set_xscale("log")
    ax[n].set_xlabel("Period [days]")
    ax[n].set_ylabel("Lomb-Scargle Power")

    # model
    n = 1
    offset = 0.5
    t_fit = np.linspace(0, 1, 100) - offset
    y_fit = ls.model(t_fit * best_period - best_period / 2, best_freq)
    ax[n].plot(t_fit * best_period, y_fit, "r-", lw=3, label="model", zorder=3)
    # fold data
    phase = ((t / best_period) % 1) - offset

    a = ax[n].scatter(phase * best_period, f, c=t, cmap=pl.get_cmap("Blues"))
    pl.colorbar(a, ax=ax[n], label=f"Time [BTJD]")
    ax[n].legend()
    if xlims is None:
        ax[n].set_xlim(-best_period / 2, best_period / 2)
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
    ax[n].set_xlabel("Phase")
    ax[n].set_ylabel("Relative flux")
    fig.tight_layout()
    return fig


def plot_odd_even(flat, tls_results, yline=None, figsize=(8, 4)):
    """
    """
    fig, axs = pl.subplots(1, 2, figsize=figsize, sharey=True)
    fold = flat.fold(period=tls_results.period, t0=tls_results.T0)

    ax = axs[0]
    fold[fold.even_mask].scatter(label="even", ax=ax)
    if yline is not None:
        ax.axhline(yline, 0, 1, lw=2, ls="--", c="k")

    ax = axs[1]
    fold[fold.odd_mask].scatter(label="odd", ax=ax)
    if yline is not None:
        ax.axhline(yline, 0, 1, lw=2, ls="--", c="k")
    return fig


def plot_rdp_pmrv(
    df,
    target_gaia_id=None,
    match_id=True,
    df_target=None,
    target_label=None,
    figsize=(10, 10),
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
            errmsg = f"Given cluster does not contain the target gaia id [{target_gaia_id}]"
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
            errmsg = f"Given cluster does not contain the target gaia id [{target_gaia_id}]"
            assert sum(idx) > 0, errmsg
            ax[n].axvline(
                df.loc[idx, par].values[0],
                0,
                1,
                c="k",
                ls="--",
                label=target_label,
            )
        else:
            assert df_target is not None, "provide df_target"
            ax[n].axvline(
                df_target[par], 0, 1, c="k", ls="--", label=target_label
            )

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
            errmsg = f"Given cluster does not contain the target gaia id [{target_gaia_id}]"
            assert sum(idx) > 0, errmsg
            ax[n].plot(
                df.loc[idx, x],
                df.loc[idx, y],
                marker=r"$\star$",
                c="y",
                ms="25",
            )
        else:
            assert df_target is not None, "provide df_target"
            ax[n].plot(
                df_target[x], df_target[y], marker=r"$\star$", c="y", ms="25"
            )
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
                    df.loc[idx, par].values[0],
                    0,
                    1,
                    c="k",
                    ls="--",
                    label=target_label,
                )
            else:
                ax[n].axvline(
                    df_target[par], 0, 1, c="k", ls="--", label=target_label
                )
        ax[n].set_xlabel("RV [km/s]")
        text = len(df[par].dropna())
        ax[n].text(
            0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes
        )
    except Exception as e:
        print(e)
        # catalog_name = df.Cluster.unique()()
        raise ValueError(
            f"radial_velocity is not available"
        )  # in {catalog_name}
    return fig


def plot_xyz_uvw(
    df,
    target_gaia_id=None,
    match_id=True,
    df_target=None,
    verbose=True,
    figsize=(12, 8),
):
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

    See also https://arxiv.org/pdf/1707.00697.pdf which estimates Sun's
    (U,V,W) = (9.03, 255.26, 7.001)
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
                    df.loc[idx, i],
                    df.loc[idx, j],
                    marker=r"$\star$",
                    c="y",
                    ms="25",
                )
            else:
                assert df_target is not None, "provide df_target"
                ax[n].plot(
                    df_target[i],
                    df_target[j],
                    marker=r"$\star$",
                    c="y",
                    ms="25",
                )
        # df.plot.scatter(x=i, y=j, ax=ax[n])
        ax[n].scatter(df[i], df[j], marker="o")
        ax[n].set_xlabel(i + " [pc]")
        ax[n].set_ylabel(j + " [pc]")
        text = len(df[[i, j]].dropna())
        ax[n].text(
            0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes
        )
        n += 1

    n = 3
    for (i, j) in itertools.combinations(["U", "V", "W"], r=2):
        if target_gaia_id is not None:
            idx = df.source_id.astype(int).isin([target_gaia_id])
            if match_id:
                errmsg = f"Given cluster does not contain the target gaia id [{target_gaia_id}]"
                assert sum(idx) > 0, errmsg
                ax[n].plot(
                    df.loc[idx, i],
                    df.loc[idx, j],
                    marker=r"$\star$",
                    c="y",
                    ms="25",
                )
            else:
                ax[n].plot(
                    df_target[i],
                    df_target[j],
                    marker=r"$\star$",
                    c="y",
                    ms="25",
                )
        # df.plot.scatter(x=i, y=j, ax=ax[n])
        ax[n].scatter(df[i], df[j], marker="o")
        ax[n].set_xlabel(i + " [km/s]")
        ax[n].set_ylabel(j + " [km/s]")
        text = len(df[[i, j]].dropna())
        ax[n].text(
            0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes
        )
        n += 1

    return fig


def plot_cmd(
    df,
    target_gaia_id=None,
    match_id=True,
    df_target=None,
    target_label=None,
    figsize=(8, 8),
    estimate_color=False,
    ax=None,
):
    """Plot color-magnitude diagram using absolute G magnitude and dereddened Bp-Rp from Gaia photometry

    Parameters
    ----------
    df : pd.DataFrame
        cluster member properties
    match_id : bool
        checks if target gaiaid in df
    df_target : pd.Series
        info of target
    estimate_color : bool
        estimate absolute/dereddened color from estimated excess

    Returns
    -------
    ax : axis
    """
    assert len(df) > 0, "df is empty"
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=figsize, constrained_layout=True)

    df["distance"] = Distance(parallax=df["parallax"].values * u.mas).pc
    # compute absolute Gmag
    df["abs_gmag"] = get_absolute_gmag(
        df["phot_g_mean_mag"], df["distance"], df["a_g_val"]
    )
    # compute intrinsic color index
    if estimate_color:
        df["bp_rp0"] = get_absolute_color_index(
            df["a_g_val"], df["phot_bp_mean_mag"], df["phot_rp_mean_mag"]
        )
    else:
        df["bp_rp0"] = df["bp_rp"] - df["e_bp_min_rp_val"]

    if target_gaia_id is not None:
        idx = df.source_id.astype(int).isin([target_gaia_id])
        if match_id:
            errmsg = f"Given cluster catalog does not contain the target gaia id [{target_gaia_id}]"
            assert sum(idx) > 0, errmsg
            ax.plot(
                df.loc[idx, "bp_rp0"],
                df.loc[idx, "abs_gmag"],
                marker=r"$\star$",
                c="y",
                ms="25",
                label=target_label,
            )
        else:
            assert df_target is not None, "provide df_target"
            df_target["distance"] = Distance(
                parallax=df_target["parallax"] * u.mas
            ).pc
            # compute absolute Gmag
            df_target["abs_gmag"] = get_absolute_gmag(
                df_target["phot_g_mean_mag"],
                df_target["distance"],
                df_target["a_g_val"],
            )
            # compute intrinsic color index
            if estimate_color:
                df_target["bp_rp0"] = get_absolute_color_index(
                    df_target["a_g_val"],
                    df_target["phot_bp_mean_mag"],
                    df_target["phot_rp_mean_mag"],
                )
            else:
                df_target["bp_rp0"] = (
                    df_target["bp_rp"] - df_target["e_bp_min_rp_val"]
                )
            ax.plot(
                df_target["bp_rp0"],
                df_target["abs_gmag"],
                marker=r"$\star$",
                c="y",
                ms="25",
                label=target_label,
            )
        if target_label is not None:
            ax.legend(loc="best")
    # df.plot.scatter(ax=ax, x="bp_rp", y="abs_gmag", marker=".")
    ax.scatter(df["bp_rp0"], df["abs_gmag"], marker=".")
    ax.set_xlabel(r"$G_{BP} - G_{RP}$ [mag]", fontsize=16)
    ax.invert_yaxis()
    ax.set_ylabel(r"$G$ [mag]", fontsize=16)

    text = len(df[["bp_rp0", "abs_gmag"]].dropna())
    ax.text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax.transAxes)
    return ax


def plot_hrd(
    df,
    target_gaia_id=None,
    match_id=True,
    df_target=None,
    target_label=None,
    figsize=(8, 8),
    yaxis="lum_val",
    xaxis="teff_val",
    ax=None,
):
    """Plot HR diagram using luminosity and Teff

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

    Returns
    -------
    ax : axis
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
                df.loc[idx, xaxis],
                df.loc[idx, yaxis],
                marker=r"$\star$",
                c="y",
                ms="25",
                label=target_label,
            )
        else:
            assert df_target is not None, "provide df_target"
            df_target["distance"] = Distance(
                parallax=df_target["parallax"] * u.mas
            ).pc
            ax.loglog(
                df_target[xaxis],
                df_target[yaxis],
                marker=r"$\star$",
                c="y",
                ms="25",
                label=target_label,
            )
        if target_label is not None:
            ax.legend(loc="best")
    # df.plot.scatter(ax=ax, x="bp_rp", y="abs_gmag", marker=".")
    ax.scatter(df[xaxis], df[yaxis], marker=".")
    ax.set_ylabel(r"$\log(L/L_{\odot})$", fontsize=16)
    ax.invert_xaxis()
    ax.set_xlabel(r"$\log(T_{\rm{eff}}$/K)", fontsize=16)
    text = len(df[[xaxis, yaxis]].dropna())
    ax.text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax.transAxes)
    return ax


def plot_pdc_sap_comparison(toiid):
    toi = get_toi(toi=toiid, verbose=False)
    period = toi["Period (days)"].values[0]
    t0 = toi["Epoch (BJD)"].values[0]
    tic = toi["TIC ID"].values[0]

    lcf = lk.search_lightcurvefile(f"TIC {tic}", mission="TESS").download()
    if lcf is not None:
        sap = lcf.SAP_FLUX.normalize()
        pdcsap = lcf.PDCSAP_FLUX.normalize()

        ax = sap.bin(11).fold(period=period, t0=t0).scatter(label="SAP")
        ax = (
            pdcsap.bin(11)
            .fold(period=period, t0=t0)
            .scatter(ax=ax, label="PDCSAP")
        )
        # ax.set_xlim(-0.1,0.1)
        ax.set_title(f"TOI {toiid}")
    return lcf, ax


def plot_hrd_spectral_types(**plot_kwargs):
    """
    """
    df = get_mamajek_table()
    fig, ax = pl.subplots(1, 1, **plot_kwargs)
    classes = []
    for idx, g in df.assign(SpT2=df["#SpT"].apply(lambda x: x[0])).groupby(
        by="SpT2"
    ):
        classes.append(idx)
        x = g["logT"].astype(float)
        y = g["logL"].astype(float)
        pl.plot(x, y, label=idx)
    pl.ylabel(r"$\log_{10}$ (L/L$_{\odot}$)")
    pl.xlabel(r"$\log_{10}$ (T$_{\rm{eff}}$/K)")
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
    figsize=(10, 10),
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
        xs=df[idx].x,
        ys=df[idx].y,
        zs=df[idx].z,
        marker=r"$\star$",
        c="r",
        s=300,
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
    ax.axhline(depth, 0, 1, c="k", ls="--", label="TESS depth")
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

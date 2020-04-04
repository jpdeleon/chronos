# -*- coding: utf-8 -*-

r"""
classes for plotting cluster properties
"""
# Import standard library
import sys
import os
from time import time as timer
import logging
import traceback

# Import modules
# from matplotlib.figure import Figure
# from matplotlib.image import AxesImage
# from loguru import logger
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
import lightkurve as lk
from scipy.ndimage import zoom
from transitleastsquares import transitleastsquares as tls

# from transitleastsquares import final_T0_fit
from astropy.coordinates import Angle, SkyCoord, Distance
from astropy.visualization import ZScaleInterval
from astroquery.mast import Catalogs
from astropy.wcs import WCS
import astropy.units as u
from astropy.stats import sigma_clip
from astroplan.plots import plot_finder_image
from astropy.timeseries import LombScargle
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from wotan import flatten
from wotan import t14 as estimate_transit_duration
import deepdish as dd
from adjustText import adjust_text

# Import from package
from chronos.star import Star
from chronos.gls import Gls
from chronos.cluster import (
    ClusterCatalog,
    Cluster,
    plot_xyz_3d,
    plot_hrd,
    plot_cmd,
    plot_rdp_pmrv,
    plot_xyz_uvw,  # with plot
)
from chronos.lightcurve import ShortCadence, LongCadence
from chronos.utils import (
    get_toi,
    get_tois,
    get_mamajek_table,
    parse_aperture_mask,
    is_point_inside_mask,
    is_gaiaid_in_cluster,
    get_fluxes_within_mask,
    get_rotation_period,
    get_transit_mask,
    bin_data,
    get_phase,
    detrend,
)

TESS_pix_scale = 21 * u.arcsec  # /pix

log = logging.getLogger(__name__)

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
    "plot_tql",
]


def plot_tql(
    gaiaid=None,
    toiid=None,
    ticid=None,
    coords=None,
    name=None,
    sector=None,
    search_radius=3,
    cadence="short",
    lctype=None,  # custom, pdcsap, sap, custom
    sap_mask=None,
    aper_radius=1,
    threshold_sigma=5,
    percentile=90,
    cutout_size=(12, 12),
    quality_bitmask="default",
    apply_data_quality_mask=False,
    flatten_method="biweight",
    window_length=0.5,  # deprecated for lk's flatten in ncadences
    Porb_limits=None,
    use_star_priors=False,
    edge_cutoff=0.1,
    run_gls=False,
    find_cluster=True,
    savefig=False,
    savetls=False,
    savegls=False,
    outdir=".",
    nearby_gaia_radius=120,  # arcsec
    bin_hr=None,
    tpf_cmap="viridis",
    verbose=True,
    clobber=False,
):
    """
    Parameters
    ----------
    cadence : str
        short, long
    lctype : str
        short=(pdcsap, sap, custom); long=(custom, cdips)
    sap_mask : str
        short=pipeline; long=square,round,threshold,percentile
    aper_radius : int
        used for square or round sap_mask (default=1 pix)
    percentile : float
        used for percentile sap_mask (default=90)
    quality_bitmask : str
        none, [default], hard, hardest; See
        https://github.com/KeplerGO/lightkurve/blob/master/lightkurve/utils.py#L135
    flatten_method : str
        wotan flatten method; See:
        https://wotan.readthedocs.io/en/latest/Interface.html#module-flatten.flatten
    window_length : float
        length in days of the filter window (default=0.5; overridden by use_star_priors)
    Porb_limits : tuple
        orbital period search limits for TLS (default=None)
    use_star_priors : bool
        priors to compute t14 for detrending in wotan,
        limb darkening in tls
    edge_cutoff : float
        length in days to be cut off each edge of lightcurve (default=0.1)
    bin_hr : float
        bin size in hours of folded lightcurves
    run_gls : bool
        run Generalized Lomb Scargle (default=False)
    find_cluster : bool
        find if target is in cluster (default=False)
    Notes:
    * removes scattered light subtraction + TESSPld
    * uses wotan's biweight to flatten lightcurve
    * uses TLS to search for transit signals

    TODO:
    * rescale x-axis of phase-folded lc in days
    * add phase offset in lomb scargle plot
    """
    start = timer()
    if Porb_limits is not None:
        # assert isinstance(Porb_limits, list)
        assert len(Porb_limits) == 2, "period_min, period_max"
        Porb_min = Porb_limits[0] if Porb_limits[0] > 0.1 else None
        Porb_max = Porb_limits[1] if Porb_limits[1] > 1 else None
    else:
        Porb_min, Porb_max = None, None

    if coords is not None:
        errmsg = "coords should be a tuple (ra dec)"
        assert len(coords) == 2, errmsg
        if len(coords[0].split(":")) == 3:
            target_coord = SkyCoord(
                ra=coords[0], dec=coords[1], unit=("hourangle", "degree")
            )
        elif len(coords[0].split(".")) == 2:
            target_coord = SkyCoord(ra=coords[0], dec=coords[1], unit="degree")
        else:
            raise ValueError("cannot decode coord input")
    else:
        target_coord = None
    try:
        if cadence == "long":
            sap_mask = "square" if sap_mask is None else sap_mask
            lctype = "custom" if lctype is None else lctype
            assert lctype in ["custom", "cdips"]
            alpha = 0.5
            lightcurve = LongCadence(
                gaiaDR2id=gaiaid,
                toiid=toiid,
                ticid=ticid,
                name=name,
                ra_deg=target_coord.ra.deg if target_coord else None,
                dec_deg=target_coord.dec.deg if target_coord else None,
                sector=sector,
                search_radius=search_radius,
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
            bin_hr = 4 if bin_hr is None else bin_hr
            # cad = np.median(np.diff(time))
            cad = 30 / 60 / 24
        elif cadence == "short":
            sap_mask = "pipeline" if sap_mask is None else sap_mask
            lctype = "pdcsap" if lctype is None else lctype
            assert lctype in ["pdcsap", "sap", "custom"]
            alpha = 0.1
            lightcurve = ShortCadence(
                gaiaDR2id=gaiaid,
                toiid=toiid,
                ticid=ticid,
                ra_deg=target_coord.ra.deg if target_coord else None,
                dec_deg=target_coord.dec.deg if target_coord else None,
                name=name,
                sector=sector,
                search_radius=search_radius,
                sap_mask=sap_mask,
                aper_radius=aper_radius,
                threshold_sigma=threshold_sigma,
                percentile=percentile,
                quality_bitmask=quality_bitmask,
                apply_data_quality_mask=apply_data_quality_mask,
                verbose=verbose,
                clobber=clobber,
            )
            bin_hr = 0.5 if bin_hr is None else bin_hr
            cad = 2 / 60 / 24
        else:
            raise ValueError("Use cadence=(long, short).")
        if verbose:
            print(f"Analyzing {cadence} cadence data with {sap_mask} mask")
        l = lightcurve
        if l.gaia_params is None:
            _ = l.query_gaia_dr2_catalog(return_nearest_xmatch=True)
        if l.tic_params is None:
            _ = l.query_tic_catalog(return_nearest_xmatch=True)
        if not l.validate_gaia_tic_xmatch():
            raise ValueError("Gaia TIC cross-match failed")

        # +++++++++++++++++++++ raw lc
        if lctype == "custom":
            # tpf is also called to make custom lc
            lc = l.make_custom_lc()
        elif lctype == "pdcsap":
            # just downloads lightcurvefile
            lc = l.get_lc(lctype)
        elif lctype == "sap":
            # just downloads lightcurvefile;
            lc = l.get_lc(lctype)
        elif lctype == "cdips":
            #  just downloads fits file
            lc = l.get_cdips_lc()
            l.aper_mask = l.cdips.get_aper_mask_cdips()
        else:
            errmsg = "use lctype=[custom,sap,pdcsap,cdips]"
            raise ValueError(errmsg)

        if (outdir is not None) & (not os.path.exists(outdir)):
            os.makedirs(outdir)

        fig, axs = pl.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
        axs = axs.flatten()

        # +++++++++++++++++++++ax: Raw + trend
        ax = axs[0]
        lc = lc.normalize().remove_nans().remove_outliers()
        flat, trend = lc.flatten(
            window_length=101, return_trend=True
        )  # flat and trend here are just place-holder
        time, flux = lc.time, lc.flux
        if use_star_priors:
            # for wotan and tls.power
            Rstar = (
                l.tic_params["rad"] if l.tic_params["rad"] is not None else 1.0
            )
            Mstar = (
                l.tic_params["mass"]
                if l.tic_params["mass"] is not None
                else 1.0
            )
            Porb = 10  # TODO: arbitrary default!
            tdur = estimate_transit_duration(
                R_s=Rstar, M_s=Mstar, P=Porb, small_planet=True
            )
            window_length = tdur * 3  # overrides default

        else:
            Rstar, Mstar = 1.0, 1.0

        wflat, wtrend = flatten(
            time,  # Array of time values
            flux,  # Array of flux values
            method=flatten_method,
            window_length=window_length,  # The length of the filter window in units of ``time``
            edge_cutoff=edge_cutoff,
            break_tolerance=0.1,  # Split into segments at breaks longer than that
            return_trend=True,
            cval=5.0,  # Tuning parameter for the robust estimators
        )
        # f > np.median(f) + 5 * np.std(f)
        idx = sigma_clip(wflat, sigma_lower=7, sigma_upper=3).mask
        # replace flux values with that from wotan
        flat = flat[~idx]
        trend = trend[~idx]
        trend.flux = wtrend[~idx]
        flat.flux = wflat[~idx]
        _ = lc.scatter(ax=ax, label="raw")
        trend.plot(ax=ax, label="trend", lw=1, c="r")

        # +++++++++++++++++++++ax2 Lomb-scargle periodogram
        ax = axs[1]
        baseline = int(time[-1] - time[0])
        Prot_max = baseline / 2

        # detrend lc
        dlc = detrend(lc, break_tolerance=10)
        ls = LombScargle(dlc.time, dlc.flux)
        frequencies, powers = ls.autopower(
            minimum_frequency=1.0 / Prot_max, maximum_frequency=1.0  # 1 day
        )
        periods = 1.0 / frequencies
        idx = np.argmax(powers)
        best_freq = frequencies[idx]
        best_period = 1.0 / best_freq
        ax.plot(periods, powers, "k-")
        ax.axvline(
            best_period, 0, 1, ls="--", c="r", label=f"peak={best_period:.2f}"
        )
        ax.legend(title="Rotation period [d]")
        ax.set_xscale("log")
        ax.set_xlabel("Period [days]")
        ax.set_ylabel("Lomb-Scargle Power")

        # +++++++++++++++++++++ax phase-folded at rotation period + sinusoidal model
        ax = axs[2]
        offset = 0.5
        t_fit = np.linspace(0, 1, 100) - offset
        y_fit = ls.model(t_fit * best_period - best_period / 2, best_freq)
        ax.plot(
            t_fit * best_period,
            y_fit,
            "r-",
            lw=3,
            label="sine model",
            zorder=3,
        )
        phase = ((time / best_period) % 1) - offset

        a = ax.scatter(
            phase * best_period,
            flux,
            c=time,
            label="folded at  Prot",
            cmap=pl.get_cmap("Blues"),
        )
        pl.colorbar(a, ax=ax, label=f"Time [BTJD]")
        ax.legend()
        ax.set_xlim(-best_period / 2, best_period / 2)
        ax.set_ylabel("Normalized Flux")
        ax.set_xlabel("Phase [days]")
        # fig.suptitle(title)

        # +++++++++++++++++++++ax5: TLS periodogram
        ax = axs[4]
        period_min = 0.1 if Porb_min is None else Porb_min
        period_max = baseline / 2 if Porb_max is None else Porb_max
        tls_results = tls(
            flat.time, flat.flux, flat.flux_err  # somewhat improves SDE
        ).power(
            R_star=Rstar,  # 0.13-3.5 default
            R_star_max=Rstar + 0.1 if Rstar > 3.5 else 3.5,
            M_star=Mstar,  # 0.1-1
            M_star_max=Mstar + 0.1 if Mstar > 1.0 else 1.0,
            period_min=period_min,  # Roche limit default
            period_max=period_max,
            n_transits_min=2,  # default
        )

        label = f"peak={tls_results.period:.3}"
        ax.axvline(tls_results.period, alpha=0.4, lw=3, label=label)
        ax.set_xlim(np.min(tls_results.periods), np.max(tls_results.periods))

        for i in range(2, 10):
            higher_harmonics = i * tls_results.period
            if period_min <= higher_harmonics <= period_max:
                ax.axvline(
                    higher_harmonics, alpha=0.4, lw=1, linestyle="dashed"
                )
            lower_harmonics = tls_results.period / i
            if period_min <= lower_harmonics <= period_max:
                ax.axvline(
                    lower_harmonics, alpha=0.4, lw=1, linestyle="dashed"
                )
        ax.set_ylabel(r"Transit Least Squares SDE")
        ax.set_xlabel("Period (days)")
        ax.plot(tls_results.periods, tls_results.power, color="black", lw=0.5)
        ax.set_xlim(period_min, period_max)
        # do not show negative SDE
        y1, y2 = ax.get_ylim()
        y1 = 0 if y1 < 0 else y1
        ax.set_ylim(y1, y2)
        ax.legend(title="Orbital period [d]")

        # +++++++++++++++++++++++ax4 : flattened lc
        ax = axs[3]
        flat.scatter(ax=ax, label="flat", zorder=1)
        # binned phase folded lc
        nbins = int(round(bin_hr / 24 / cad))
        # transit mask
        tmask = get_transit_mask(
            flat, tls_results.period, tls_results.T0, tls_results.duration * 24
        )
        flat[tmask].scatter(ax=ax, label="transit", c="r", alpha=0.5, zorder=1)

        # +++++++++++++++++++++ax6: phase-folded at orbital period
        ax = axs[5]
        # binned phase folded lc
        fold = flat.fold(period=tls_results.period, t0=tls_results.T0)
        fold.scatter(
            ax=ax, c="k", alpha=alpha, label="folded at Porb", zorder=1
        )
        fold.bin(nbins).scatter(
            ax=ax, s=30, label=f"{bin_hr}-hr bin", zorder=2
        )

        # TLS transit model
        ax.plot(
            tls_results.model_folded_phase - offset,
            tls_results.model_folded_model,
            color="red",
            zorder=3,
            label="TLS model",
        )
        ax.set_xlabel("Phase")
        ax.set_ylabel("Relative flux")
        width = tls_results.duration / tls_results.period
        ax.set_xlim(-width * 1.5, width * 1.5)
        ax.legend()

        # +++++++++++++++++++++ax: odd-even
        ax = axs[6]
        yline = tls_results.depth
        fold.scatter(ax=ax, c="k", alpha=alpha, label="_nolegend_", zorder=1)
        fold[fold.even_mask].bin(nbins).scatter(
            label="even", s=30, ax=ax, zorder=2
        )
        ax.plot(
            tls_results.model_folded_phase - offset,
            tls_results.model_folded_model,
            color="red",
            zorder=3,
            label="TLS model",
        )
        ax.axhline(yline, 0, 1, lw=2, ls="--", c="k")
        fold[fold.odd_mask].bin(nbins).scatter(
            label="odd", s=30, ax=ax, zorder=3
        )
        ax.axhline(yline, 0, 1, lw=2, ls="--", c="k")
        ax.set_xlim(-width * 1.5, width * 1.5)
        ax.legend()

        # +++++++++++++++++++++ax7: tpf
        ax = axs[7]
        if cadence == "short":
            if l.tpf is None:
                # e.g. pdcsap, sap
                tpf = l.get_tpf()
            else:
                # e.g. custom
                tpf = l.tpf
        else:
            if l.tpf_tesscut is None:
                # e.g. cdips
                tpf = l.get_tpf_tesscut()
            else:
                # e.g. custom
                tpf = l.tpf_tesscut

        if (l.gaia_sources is None) or (nearby_gaia_radius != 120):
            _ = l.query_gaia_dr2_catalog(radius=nearby_gaia_radius)
        # _ = plot_orientation(tpf, ax)
        _ = plot_gaia_sources_on_tpf(
            tpf=tpf,
            target_gaiaid=l.gaiaid,
            gaia_sources=l.gaia_sources,
            kmax=1,
            depth=1 - tls_results.depth,
            sap_mask=l.sap_mask,
            aper_radius=l.aper_radius,
            threshold_sigma=l.threshold_sigma,
            percentile=l.percentile,
            cmap=tpf_cmap,
            dmag_limit=8,
            ax=ax,
        )

        if l.contratio is None:
            # also computed in make_custom_lc()
            l.aper_mask = parse_aperture_mask(
                tpf,
                sap_mask=l.sap_mask,
                aper_radius=l.aper_radius,
                percentile=l.percentile,
                threshold_sigma=l.threshold_sigma,
            )
            fluxes = get_fluxes_within_mask(tpf, l.aper_mask, l.gaia_sources)
            l.contratio = sum(fluxes) - 1  # c.f. l.tic_params.contratio

        # +++++++++++++++++++++ax: summary
        tp = l.tic_params
        ax = axs[8]
        Rp = tls_results["rp_rs"] * tp["rad"] * u.Rsun.to(u.Rearth)
        # np.sqrt(tls_results["depth"]*(1+l.contratio))
        Rp_true = Rp * np.sqrt(1 + l.contratio)
        msg = "Candidate Properties\n"
        msg += "-" * 30 + "\n"
        # secs = ','.join(map(str, l.all_sectors))
        msg += f"SDE={tls_results.SDE:.2f} (sector={l.sector} in {l.all_sectors})\n"
        msg += (
            f"Period={tls_results.period:.2f}+/-{tls_results.period_uncertainty:.2f} d"
            + " " * 5
        )
        msg += f"T0={tls_results.T0:.2f} BTJD\n"
        msg += f"Duration={tls_results.duration*24:.2f} hr" + " " * 10
        msg += f"Depth={(1-tls_results.depth)*100:.2f}%\n"
        msg += f"Rp={Rp:.2f} " + r"R$_{\oplus}$" + "(diluted)" + " " * 5
        msg += f"Rp={Rp_true:.2f} " + r"R$_{\oplus}$" + "(undiluted)\n"
        msg += (
            f"Odd-Even mismatch={tls_results.odd_even_mismatch:.2f}"
            + r"$\sigma$"
        )
        msg += "\n" * 2
        msg += "Stellar Properties\n"
        msg += "-" * 30 + "\n"
        msg += f"TIC ID={int(tp['ID'])}" + " " * 5
        msg += f"Tmag={tp['Tmag']:.2f}\n"
        msg += (
            f"Rstar={tp['rad']:.2f}+/-{tp['e_rad']:.2f} "
            + r"R$_{\odot}$"
            + " " * 5
        )
        msg += (
            f"Mstar={tp['mass']:.2f}+/-{tp['e_mass']:.2f} "
            + r"M$_{\odot}$"
            + "\n"
        )
        teff = "nan" if str(tp["Teff"]).lower() == "nan" else int(tp["Teff"])
        eteff = (
            "nan" if str(tp["e_Teff"]).lower() == "nan" else int(tp["e_Teff"])
        )
        msg += f"Teff={teff}+/-{eteff} K" + " " * 5
        msg += f"logg={tp['logg']:.2f}+/-{tp['e_logg']:.2f} dex\n"
        # spectype = star.get_spectral_type()
        # msg += f"SpT: {spectype}\n"
        msg += r"$\rho$" + f"star={tp['rho']:.2f}+/-{tp['e_rho']:.2f} gcc\n"
        msg += f"Contamination ratio={l.contratio:.2f}% (TIC={tp['contratio']:.2f}%)\n"
        ax.text(0, 0, msg, fontsize=10)
        ax.axis("off")

        if l.toiid is not None:
            fig.suptitle(f"TOI {l.toiid} | TIC {l.ticid} (sector {l.sector})")
        else:
            fig.suptitle(f"TIC {l.ticid} (sector {l.sector})")
        # fig.tight_layout()
        if run_gls:
            if verbose:
                print("Running GLS pipeline")
            data = (flat.time, flat.flux, flat.flux_err)
            gls = Gls(data, Pbeg=1, verbose=True)
            # show plot if not saved
            fig2 = gls.plot(block=~savefig, figsize=(10, 8))
        if find_cluster:
            is_gaiaid_in_cluster(
                l.gaiaid, catalog_name="Bouma2019", verbose=True
            )
            # function prints output
        end = timer()
        msg = ""
        if savefig:
            fp = os.path.join(
                outdir, f"tic{l.ticid}_s{l.sector}_{lctype}_{cadence[0]}c"
            )
            fig.savefig(fp + ".png", bbox_inches="tight")
            msg += f"Saved: {fp}.png\n"
            if run_gls:
                fig2.savefig(fp + "_gls.png", bbox_inches="tight")
                msg += f"Saved: {fp}_gls.png\n"
        if savetls:
            tls_results["gaiaid"] = l.gaiaid
            tls_results["ticid"] = l.ticid
            dd.io.save(fp + "_tls.h5", tls_results)
            msg += f"Saved: {fp}_tls.h5\n"

        msg += f"#----------Runtime: {end-start:.2f} s----------#\n"
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
            print(f"Line : {trace[1]}")
            print(f"Func : {trace[2]}")
            print(f"Message : {trace[3]}")


def plot_cluster_map(
    target_coord=None,
    catalog_name="Bouma2019",
    cluster_name=None,
    offset=10,
    ax=None,
):
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
    xdeg = (nx * TESS_pix_scale).to(u.arcmin)
    ydeg = (ny * TESS_pix_scale).to(u.arcmin)
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
    if method == "lombscargle":
        ls = LombScargle(time, flux)
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


def plot_interactive(catalog_name="CantatGaudin2020", parallax_cut=2):
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

    print("import altair; altair.notebook()")

    cc = ClusterCatalog(verbose=False)
    # get Bouma catalog
    df0 = cc.query_catalog(catalog_name=catalog_name, return_members=False)
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
                # "Cluster:Nstars",
                "distance:Q",
                "parallax:Q",
                "pmra:Q",
                "pmdec:Q",
                # "count:Q",
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
    df2 = cc.query_catalog(catalog_name=catalog_name, return_members=True)
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

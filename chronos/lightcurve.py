# -*- coding: utf-8 -*-

r"""
Module for manipulating TESS light curve.
See k2 module for K2 light curve.

Tess class is a holder for methods applicable for TESS targets.
LongCadence and ShortCadence classes provide the core TESS light curve functionalities.
"""
# Import standard library
from os.path import join, exists
import logging

# Import library
import getpass
import numpy as np
import matplotlib.pyplot as pl
import astropy.units as u

from scipy.signal import detrend as linear_detrend
from astropy.timeseries import LombScargle
from astropy.io import fits
import lightkurve as lk
from wotan import flatten
from transitleastsquares import transitleastsquares

# Import from package
from chronos.config import DATA_PATH
from chronos.target import Target
from chronos.tpf import Tpf, FFI_cutout
from chronos.cdips import CDIPS
from chronos.pathos import PATHOS
from chronos.qlp import QLP
from chronos.plot import plot_tls, plot_odd_even, plot_aperture_outline
from chronos.utils import (
    get_all_sectors,
    remove_bad_data,
    parse_aperture_mask,
    get_fluxes_within_mask,
    get_transit_mask,
    detrend,
    get_toi,
)
from chronos.constants import TESS_TIME_OFFSET

user = getpass.getuser()
MISSION = "TESS"
fitsoutdir = join("/home", user, "data/transit")

pl.style.use("default")
log = logging.getLogger(__name__)

__all__ = ["Tess", "LongCadence", "ShortCadence"]


class Tess(Target):
    """
    Provides static methods applicable for TESS targets
    """

    def __ini__(
        self,
        name=None,
        toiid=None,
        ticid=None,
        epicid=None,
        gaiaDR2id=None,
        ra_deg=None,
        dec_deg=None,
        mission="tess",
        search_radius=3,
    ):
        super().__init__(
            name=name,
            toiid=toiid,
            ticid=ticid,
            epicid=epicid,
            gaiaDR2id=gaiaDR2id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            search_radius=search_radius,
            mission=mission,
        )

    # @staticmethod
    def plot_pdc_sap_comparison(self, period=None, t0=None):
        """
        """
        sc = ShortCadence(toiid=self.toiid, ticid=self.ticid)
        sap = sc.lc_sap
        pdcsap = sc.lc_pdcsap
        if (period is None) | (t0 is None):
            if self.toi_params is not None:
                period = self.toi_period
                t0 = self.toi_epoch
        else:
            raise ValueError("Provide `period` and `t0`.")
        ax = sap.bin(11).fold(period=period, t0=t0).scatter(label="SAP")
        _ = (
            pdcsap.bin(11)
            .fold(period=period, t0=t0)
            .scatter(ax=ax, label="PDCSAP")
        )
        # ax.set_xlim(-0.1,0.1)
        ax.set_title(f"TOI {self.toiid} (sector {sap.sector})")
        return ax

    # @staticmethod
    def get_lightcurves(
        self,
        pipeline="pdcsap",
        cadence="short",
        sectors=None,
        remove_outliers=False,
        quality_bitmask=None,
    ):
        """
        download all lightcurves in the given sectors
        """
        if sectors is None:
            all_sectors = self.all_sectors
        else:
            all_sectors = sectors

        for n, sector in enumerate(all_sectors):
            if pipeline == "pdcsap":
                l = ShortCadence(
                    ticid=self.ticid, sector=sector, verbose=False
                )
                lc = l.get_lc()
            else:
                errmsg = "pdcsap is only currently available"
                raise NotImplementedError(errmsg)

            if quality_bitmask == "hard":
                lc = lc[(lc.quality == 0) | np.isnan(lc.quality)]

            if remove_outliers:
                lc, mask = lc.remove_outliers(
                    sigma_upper=3, sigma_lower=10, return_mask=True
                )

            if n == 0:
                lcs = lc.copy()
            else:
                lcs = lcs.append(lc)
            print(
                f"{sector}: cdpp={lc.estimate_cdpp():.2f}, std={lc.flux.std():.2f}"
            )

        lcs.sector = all_sectors
        return lcs


class LongCadence(FFI_cutout):
    """
    handles lightcurve creation and manipulation for TESS long cadence data
    using `FFI_cutout`. Inherits FFI_cutout class.
    """

    def __init__(
        self,
        sector=None,
        name=None,
        toiid=None,
        ticid=None,
        epicid=None,
        gaiaDR2id=None,
        ra_deg=None,
        dec_deg=None,
        mission="tess",
        search_radius=3,
        sap_mask="square",
        aper_radius=1,
        threshold_sigma=5,
        percentile=95,
        cutout_size=(15, 15),
        quality_bitmask="default",
        apply_data_quality_mask=False,
        calc_fpp=False,
        clobber=True,
        verbose=True,
    ):
        """
        Attributes
        ----------
        sap_mask : str
            aperture mask shape (default=square)
        aper_radius : int
            aperture radius
        threshold_sigma : float
            threshold sigma above median flux
        percentile : float
            percentile of flux
        quality_bitmask : str
            (default=default)
            https://github.com/KeplerGO/lightkurve/blob/master/lightkurve/utils.py#L210
        apply_data_quality_mask : bool (default=False)
            remove bad data identified in TESS Data Release notes

        """
        super().__init__(
            name=name,
            toiid=toiid,
            ticid=ticid,
            epicid=epicid,
            gaiaDR2id=gaiaDR2id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            sector=sector,
            search_radius=search_radius,
            sap_mask=sap_mask,
            aper_radius=aper_radius,
            threshold_sigma=threshold_sigma,
            percentile=percentile,
            cutout_size=cutout_size,
            quality_bitmask=quality_bitmask,
            apply_data_quality_mask=apply_data_quality_mask,
            calc_fpp=calc_fpp,
            verbose=verbose,
            clobber=clobber,
        )
        self.corrector = None
        self.lc_custom = None
        self.lc_custom_raw = None
        self.lc_cdips = None
        self.lc_pathos = None
        self.contratio = None
        self.cdips = None
        self.pathos = None
        self.qlp = None
        self.tls_results = None
        _ = self.make_custom_lc()

        if self.verbose:
            print(f"Using {self.mission.upper()} long cadence.\n")

    def make_custom_lc(
        self,
        sector=None,
        tpf_size=None,
        sap_mask=None,
        aper_radius=None,
        percentile=None,
        threshold_sigma=None,
        use_pld=True,
        pixel_components=3,
        spline_n_knots=100,
        spline_degree=3,
        background_mask=None,
        pca_nterms=5,
        with_offset=True,
    ):
        """
        create a custom lightcurve based on this tutorial:
        https://docs.lightkurve.org/tutorials/04-how-to-remove-tess-scattered-light-using-regressioncorrector.html

        Parameters
        ----------
        sector : int or str
            specific sector or all
        cutout_size : tuple
            tpf cutout size
        aper_radius: int
            aperture mask radius
        percentile: float
            aperture mask percentile
        threshold_sigma: float
            aperture mask threshold [sigma]
        method : float
            PLD (default)

        Returns
        -------
        corrected_lc : lightkurve object
        """
        if self.verbose:
            print("Using lightcurve with custom aperture.")
        sector = sector if sector is not None else self.sector
        sap_mask = sap_mask if sap_mask else self.sap_mask
        aper_radius = aper_radius if aper_radius else self.aper_radius
        percentile = percentile if percentile else self.percentile
        threshold_sigma = (
            threshold_sigma if threshold_sigma else self.threshold_sigma
        )
        cutout_size = tpf_size if tpf_size else self.cutout_size

        tpf_tesscut = self.get_tpf_tesscut(
            sector=sector, cutout_size=cutout_size
        )

        self.aper_mask = parse_aperture_mask(
            tpf_tesscut,
            sap_mask=sap_mask,
            aper_radius=aper_radius,
            percentile=percentile,
            threshold_sigma=threshold_sigma,
            verbose=False,
        )

        raw_lc = tpf_tesscut.to_lightcurve(
            method="aperture", aperture_mask=self.aper_mask
        )
        # remove nans
        idx = (
            np.isnan(raw_lc.time)
            | np.isnan(raw_lc.flux)
            | np.isnan(raw_lc.flux_err)
        )
        self.tpf_tesscut = tpf_tesscut[~idx]
        self.lc_custom_raw = raw_lc[~idx]

        if use_pld:
            if self.verbose:
                print("Removing scattered light + applying PLD")
            pld = lk.TessPLDCorrector(
                self.tpf_tesscut, aperture_mask=self.aper_mask
            )
            if background_mask is None:
                background_mask = ~self.aper_mask
            corrected_lc = pld.correct(
                pixel_components=pixel_components,
                spline_n_knots=spline_n_knots,
                spline_degree=spline_degree,
                background_mask=background_mask,
            )
            self.corrector = pld
        else:
            if self.verbose:
                print("Removing scattered light")
            # Make a design matrix and pass it to a linear regression corrector
            regressors = tpf_tesscut.flux[~idx][:, ~self.aper_mask]
            dm = (
                lk.DesignMatrix(regressors, name="regressors")
                .pca(nterms=pca_nterms)
                .append_constant()
            )
            rc = lk.RegressionCorrector(raw_lc)
            self.corrector = rc
            corrected_lc = rc.correct(dm)

            # Optional: Remove the scattered light, allowing for the large offset from scattered light
            if with_offset:
                corrected_lc = (
                    raw_lc - rc.model_lc + np.percentile(rc.model_lc.flux, q=5)
                )
        lc = corrected_lc.normalize()
        self.lc_custom = lc

        # compute Contamination
        if self.gaia_sources is None:
            gaia_sources = self.query_gaia_dr2_catalog(radius=120)
        else:
            gaia_sources = self.gaia_sources
        fluxes = get_fluxes_within_mask(
            self.tpf_tesscut, self.aper_mask, gaia_sources
        )
        self.contratio = sum(fluxes) - 1
        # add method
        lc.detrend = lambda: detrend(lc)
        return lc

    def get_qlp_lc(self, sector=None, lctype="kspsap", verbose=False):
        verbose = verbose if verbose is not None else self.verbose
        sector = sector if sector is not None else self.sector
        if self.gaiaid is None:
            d = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)
            self.gaiaid = int(d.source_id)
        qlp = QLP(
            toiid=self.toiid,
            ticid=self.ticid,
            gaiaDR2id=self.gaiaid,
            sector=sector,
            lctype=lctype,
            verbose=verbose,
        )
        self.qlp = qlp
        self.lc_qlp = qlp.lc
        self.lc_qlp.targetid = self.ticid
        return qlp.lc

    def get_cdips_lc(
        self, sector=None, aper_idx=3, lctype="flux", verbose=False
    ):
        verbose = verbose if verbose is not None else self.verbose
        sector = sector if sector is not None else self.sector
        if self.gaiaid is None:
            d = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)
            self.gaiaid = int(d.source_id)
        cdips = CDIPS(
            toiid=self.toiid,
            ticid=self.ticid,
            gaiaDR2id=self.gaiaid,
            sector=sector,
            aper_idx=aper_idx,
            lctype=lctype,
            verbose=verbose,
        )
        self.cdips = cdips
        self.lc_cdips = cdips.lc
        self.lc_cdips.targetid = self.ticid
        return cdips.lc

    def get_pathos_lc(
        self, sector=None, aper_idx=4, lctype="corr", verbose=False
    ):
        verbose = verbose if verbose is not None else self.verbose
        sector = sector if sector is not None else self.sector
        if self.gaiaid is None:
            d = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)
            self.gaiaid = int(d.source_id)
        pathos = PATHOS(
            toiid=self.toiid,
            ticid=self.ticid,
            gaiaDR2id=self.gaiaid,
            sector=sector,
            aper_idx=aper_idx,
            lctype=lctype,
            verbose=verbose,
        )
        self.pathos = pathos
        self.lc_pathos = pathos.lc
        self.lc_pathos.targetid = self.ticid
        return pathos.lc

    def plot_lc_per_aperture(
        self,
        sector=None,
        kwargs={"aper_radius": 1, "percentile": 84, "threshold_sigma": 3},
        apertures=["round", "square", "percentile", "threshold"],
        return_lcs=False,
    ):
        """
        plot lightcurves with varying aperture shapes
        """
        sector = self.sector if sector is None else sector
        nrows = len(apertures)
        fig, axs = pl.subplots(
            nrows=nrows,
            ncols=2,
            figsize=(10, nrows * 2),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [3, 1], "hspace": 0, "wspace": 0},
        )
        custom_lcs = {}
        for n, sap_mask in enumerate(apertures):
            ax1 = axs[n, 0]
            lc = self.make_custom_lc(
                sector=sector, sap_mask=sap_mask, **kwargs
            )
            lc.scatter(ax=ax1, label=sap_mask)
            print(f"mask={sap_mask}; contratio={self.contratio:.2f}")
            custom_lcs[sap_mask] = lc
            if n != len(apertures) - 1:
                ax1.set_xlabel("")
                ax1.set_xticklabels("")
            if n == 0:
                ax1.set_title(f"{self.target_name} (sector {sector})")
            if self.tpf_tesscut is None:
                tpf = self.get_tpf_tesscut()
            else:
                tpf = self.tpf_tesscut
            img = np.nanmedian(self.tpf_tesscut.flux, axis=0)

            ax2 = axs[n, 1]
            ax = plot_aperture_outline(
                img, mask=self.aper_mask, imgwcs=tpf.wcs, ax=ax2
            )
            ax.axis("off")
        if return_lcs:
            return fig, custom_lcs
        else:
            return fig

    def get_flat_lc(
        self,
        lc,
        window_length=None,
        period=None,
        epoch=None,
        duration=None,
        method="biweight",
        return_trend=False,
        **wotan_kwargs,
    ):
        return _get_flat_lc(
            self=self,
            lc=lc,
            period=period,
            epoch=epoch,
            duration=duration,
            window_length=window_length,
            method=method,
            return_trend=return_trend,
            **wotan_kwargs,
        )

    def plot_trend_flat_lcs(
        self, lc, period=None, epoch=None, duration=None, binsize=10, **kwargs
    ):
        return _plot_trend_flat_lcs(
            self=self,
            lc=lc,
            period=period,
            epoch=epoch,
            duration=duration,
            binsize=binsize,
            **kwargs,
        )

    def plot_fold_lc(
        self, flat=None, period=None, epoch=None, duration=None, ax=None
    ):
        return _plot_fold_lc(
            self=self,
            flat=flat,
            period=period,
            epoch=epoch,
            duration=duration,
            ax=ax,
        )

    def run_tls(self, flat, plot=True, **tls_kwargs):
        """
        """
        tls = transitleastsquares(t=flat.time, y=flat.flux, dy=flat.flux_err)
        tls_results = tls.power(**tls_kwargs)
        self.tls_results = tls_results
        if plot:
            fig = plot_tls(tls_results)
            fig.axes[0].set_title(f"{self.target_name} (sector {flat.sector})")
            return fig

    def plot_odd_even(
        self, flat, period=None, epoch=None, duration=None, ylim=None
    ):
        """
        """
        period = self.toi_period if period is None else period
        epoch = self.toi_epoch - TESS_TIME_OFFSET if epoch is None else epoch
        duration = self.duration if duration is None else duration
        if (period is None) or (epoch is None):
            if self.tls_results is None:
                print("Running TLS")
                _ = self.run_tls(flat, plot=False)
            period = self.tls_results.period
            epoch = self.tls_results.T0
            ylim = self.tls_results.depth if ylim is None else ylim
        if ylim is None:
            ylim = 1 - self.toi_depth
        fig = plot_odd_even(
            flat, period=period, epoch=epoch, duration=duration, yline=ylim
        )
        fig.suptitle(f"{self.target_name} (sector {flat.sector})")
        return fig

    def get_transit_mask(self, lc, period=None, epoch=None, duration=None):
        """
        """
        period = self.toi_period if period is None else period
        epoch = self.toi_epoch - TESS_TIME_OFFSET if epoch is None else epoch
        duration = self.toi_duration if duration is None else duration
        tmask = get_transit_mask(
            lc, period=period, epoch=epoch, duration_hours=duration
        )
        return tmask

    @property
    def cadence(self):
        return "long"

    @staticmethod
    def plot_out_of_transit(flat, per, t0, depth):
        """
        """
        fig, axs = pl.subplots(
            3, 1, figsize=(10, 10), gridspec_kw={"hspace": 0.1}
        )
        dy = 5 if depth < 0.01 else 1.5
        ylim = (1 - dy * depth, 1 + 1.1 * depth)

        _ = _plot_fold_lc(
            flat, period=per, epoch=t0 + per / 2, duration=None, ax=axs[0]
        )
        axs[0].axhline(1 - depth, 0, 1, c="C1", ls="--")
        pl.setp(axs[0], xlim=(-0.5, 0.5), ylim=ylim)

        _ = _plot_fold_lc(
            flat, period=per, epoch=t0 + per / 2, duration=None, ax=axs[1]
        )
        axs[1].axhline(1 - depth, 0, 1, c="C1", ls="--")
        axs[1].legend("")
        pl.setp(axs[1], xlim=(-0.3, 0.3), title="", ylim=ylim)

        _ = _plot_fold_lc(
            flat, period=per, epoch=t0 + per / 2, duration=None, ax=axs[2]
        )
        axs[2].axhline(1 - depth, 0, 1, c="C1", ls="--")
        axs[2].legend("")
        pl.setp(axs[2], xlim=(-0.1, 0.1), title="", ylim=ylim)
        return fig


class ShortCadence(Tpf):
    """
    handles lightcurve creation and manipulation for TESS short cadence data
    using `lightkurve`. Inherits Tpf class.
    """

    def __init__(
        self,
        sector=None,
        name=None,
        toiid=None,
        ticid=None,
        epicid=None,
        gaiaDR2id=None,
        ra_deg=None,
        dec_deg=None,
        mission="tess",
        search_radius=3,
        sap_mask="pipeline",
        aper_radius=1,
        threshold_sigma=5,
        percentile=95,
        quality_bitmask="default",
        apply_data_quality_mask=False,
        apphot_method="aperture",  # or prf
        calc_fpp=False,
        clobber=True,
        verbose=True,
    ):
        """
        sap_mask : str
            aperture mask shape (default=pipeline)
        aper_radius : int
            if aperture radius for mask!=pipeline
        threshold_sigma : float
            threshold sigma above median flux for mask!=pipeline
        percentile : float
            percentile of flux for mask!=pipeline
        quality_bitmask : str
            (default=default)
            https://github.com/KeplerGO/lightkurve/blob/master/lightkurve/utils.py#L210
        apply_data_quality_mask : bool (default=False)
            remove bad data identified in TESS Data Release notes
        """
        super().__init__(
            name=name,
            toiid=toiid,
            ticid=ticid,
            epicid=epicid,
            gaiaDR2id=gaiaDR2id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            sector=sector,
            search_radius=search_radius,
            sap_mask=sap_mask,
            aper_radius=aper_radius,
            threshold_sigma=threshold_sigma,
            percentile=percentile,
            quality_bitmask=quality_bitmask,
            apply_data_quality_mask=apply_data_quality_mask,
            calc_fpp=calc_fpp,
            verbose=verbose,
            clobber=clobber,
        )
        self.apphot_method = apphot_method
        self.lc_custom = None
        self.lc_custom_raw = None
        self.lcf = None
        self.lc_sap = None
        self.lc_pdcsap = None
        self.contratio = None
        self.tls_results = None
        _ = self.get_lc()

        if self.verbose:
            print(f"Using {self.mission.upper()} short cadence.\n")

    def get_lc(self, lctype="pdcsap", sector=None, quality_bitmask=None):
        """
        """
        sector = sector if sector is not None else self.sector
        quality_bitmask = (
            quality_bitmask if quality_bitmask else self.quality_bitmask
        )
        if self.lcf is not None:
            # reload lcf if already in memory
            if self.lcf.sector == sector:
                lcf = self.lcf
            else:
                query_str = (
                    f"TIC {self.ticid}" if self.ticid else self.target_coord
                )
                if self.verbose:
                    print(
                        f"Searching lightcurvefile for {query_str} (sector {sector})"
                    )
                q = lk.search_lightcurvefile(
                    query_str, sector=sector, mission=MISSION
                )
                if len(q) == 0:
                    if self.verbose:
                        print(
                            f"Searching lightcurvefile for {self.target_coord.to_string()} (sector {sector})"
                        )
                    q = lk.search_lightcurvefile(
                        self.target_coord, sector=sector, mission=MISSION
                    )
                assert q is not None, "Empty result. Check long cadence."
                if self.verbose:
                    print(f"Found {len(q)} lightcurves")
                if (sector == "all") & (len(self.all_sectors) > 1):
                    lcf = q.download_all(quality_bitmask=quality_bitmask)
                else:
                    lcf = q.download(quality_bitmask=quality_bitmask)
                self.lcf = lcf
        else:
            query_str = (
                f"TIC {self.ticid}" if self.ticid else self.target_coord
            )
            if self.verbose:
                print(
                    f"Searching lightcurvefile for {query_str} (sector {sector})"
                )
            q = lk.search_lightcurvefile(
                query_str, sector=sector, mission=MISSION
            )
            if len(q) == 0:
                if self.verbose:
                    print(
                        f"Searching lightcurvefile for {self.target_coord.to_string()} (sector {sector})"
                    )
                q = lk.search_lightcurvefile(
                    self.target_coord, sector=sector, mission=MISSION
                )
            assert q is not None, "Empty result. Check long cadence."
            if self.verbose:
                print(f"Found {len(q)} lightcurves")
            if (sector == "all") & (len(self.all_sectors) > 1):
                lcf = q.download_all(quality_bitmask=quality_bitmask)
            else:
                lcf = q.download(quality_bitmask=quality_bitmask)
            self.lcf = lcf
        assert lcf is not None, "Empty result. Check long cadence."
        sap = lcf.SAP_FLUX
        pdcsap = lcf.PDCSAP_FLUX
        if isinstance(lcf, lk.LightCurveFileCollection):
            # merge multi-sector into one lc
            if len(lcf) > 1:
                sap0 = sap[0].normalize()
                sap = [sap0.append(l.normalize()) for l in sap[1:]][0]
                pdcsap0 = pdcsap[0].normalize()
                pdcsap = [pdcsap0.append(l.normalize()) for l in pdcsap[1:]][0]
            else:
                raise ValueError(
                    f"Only sector {lcf[0].sector} (in {self.all_sectors}) is available"
                )
        self.lc_sap = sap
        self.lc_pdcsap = pdcsap
        if lctype == "pdcsap":
            # add detrend method to lc instance
            pdcsap.detrend = lambda: detrend(pdcsap)
            return pdcsap.remove_nans().normalize()
        else:
            sap.detrend = lambda: detrend(sap)
            return sap.remove_nans().normalize()

    def make_custom_lc(
        self,
        sector=None,
        sap_mask=None,
        aper_radius=None,
        percentile=None,
        threshold_sigma=None,
        use_pld=True,
        pixel_components=3,
        spline_n_knots=100,
        spline_degree=3,
        background_mask=None,
        pca_nterms=5,
        with_offset=True,
    ):
        """
        create a custom lightcurve with background subtraction, based on this tutorial:
        https://docs.lightkurve.org/tutorials/04-how-to-remove-tess-scattered-light-using-regressioncorrector.html

        Parameters
        ----------
        sector : int or str
            specific sector or all
        aper_radius: int
            aperture mask radius
        percentile: float
            aperture mask percentile
        threshold_sigma: float
            aperture mask threshold [sigma]
        pca_nterms : int
            number of pca terms to use

        Returns
        -------
        corrected_lc : lightkurve object
        """
        if self.verbose:
            print("Using lightcurve with custom aperture.")
        sector = sector if sector is not None else self.sector
        sap_mask = sap_mask if sap_mask else self.sap_mask
        aper_radius = aper_radius if aper_radius else self.aper_radius
        percentile = percentile if percentile else self.percentile
        threshold_sigma = (
            threshold_sigma if threshold_sigma else self.threshold_sigma
        )
        if self.tpf is None:
            tpf, tpf_info = self.get_tpf(sector=sector, return_df=True)
        else:
            if self.tpf.sector == sector:
                tpf = self.tpf
            else:
                tpf, tpf_info = self.get_tpf(sector=sector, return_df=True)
        # Make an aperture mask and a raw light curve
        self.aper_mask = parse_aperture_mask(
            tpf,
            sap_mask=sap_mask,
            aper_radius=aper_radius,
            percentile=percentile,
            threshold_sigma=threshold_sigma,
            verbose=False,
        )
        raw_lc = tpf.to_lightcurve(
            method="aperture", aperture_mask=self.aper_mask
        )
        # remove nans
        idx = (
            np.isnan(raw_lc.time)
            | np.isnan(raw_lc.flux)
            | np.isnan(raw_lc.flux_err)
        )
        self.tpf = tpf[~idx]
        self.raw_lc = raw_lc[~idx]

        if use_pld:
            if self.verbose:
                print("Removing scattered light + applying PLD")
            pld = lk.TessPLDCorrector(self.tpf, aperture_mask=self.aper_mask)
            if background_mask is None:
                background_mask = ~self.aper_mask
            corrected_lc = pld.correct(
                pixel_components=pixel_components,
                spline_n_knots=spline_n_knots,
                spline_degree=spline_degree,
                background_mask=background_mask,
            )
            self.corrector = pld
        else:
            if self.verbose:
                print("Removing scattered light")
            # Make a design matrix and pass it to a linear regression corrector
            regressors = tpf.flux[~idx][:, ~self.aper_mask]
            dm = (
                lk.DesignMatrix(regressors, name="pixels")
                .pca(pca_nterms)
                .append_constant()
            )

            # Regression Corrector Object
            rc = lk.RegressionCorrector(self.raw_lc)
            self.corrector = rc
            corrected_lc = rc.correct(dm)

            # Optional: Remove the scattered light, allowing for the large offset from scattered light
            if with_offset:
                corrected_lc = (
                    self.raw_lc
                    - rc.model_lc
                    + np.percentile(rc.model_lc.flux, q=5)
                )
        lc = corrected_lc.normalize()
        self.lc_custom = lc
        # compute Contamination
        if self.gaia_sources is None:
            gaia_sources = self.query_gaia_dr2_catalog(
                radius=120, verbose=False
            )
        else:
            gaia_sources = self.gaia_sources
        fluxes = get_fluxes_within_mask(self.tpf, self.aper_mask, gaia_sources)
        self.contratio = sum(fluxes) - 1
        if self.tic_params is None:
            _ = self.query_tic_catalog(return_nearest_xmatch=True)
        tic_contratio = self.tic_params.contratio
        dcontratio = abs(tic_contratio - self.contratio)
        if (tic_contratio is not None) & (dcontratio > 0.5):
            print(f"contratio: {self.contratio:.2f} (TIC={tic_contratio:.2f})")

        # add method
        lc.detrend = lambda: detrend(lc)
        return lc

    def plot_lc_per_aperture(
        self,
        sector=None,
        kwargs={"aper_radius": 1, "percentile": 84, "threshold_sigma": 3},
        apertures=["pipeline", "round", "square", "percentile", "threshold"],
        return_lcs=False,
    ):
        """
        plot lightcurves with varying aperture shapes
        """
        sector = self.sector if sector is None else sector
        nrows = len(apertures)
        fig, axs = pl.subplots(
            nrows=nrows,
            ncols=2,
            figsize=(10, nrows * 2),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [3, 1], "hspace": 0, "wspace": 0},
        )
        custom_lcs = {}
        for n, sap_mask in enumerate(apertures):
            ax1 = axs[n, 0]
            lc = self.make_custom_lc(
                sector=sector, sap_mask=sap_mask, **kwargs
            )
            lc.scatter(ax=ax1, label=sap_mask)
            print(f"mask={sap_mask}; contratio={self.contratio:.2f}")
            custom_lcs[sap_mask] = lc
            if n != len(apertures) - 1:
                ax1.set_xlabel("")
                ax1.set_xticklabels("")
            if n == 0:
                ax1.set_title(f"{self.target_name} (sector {sector})")
            if self.tpf is None:
                tpf = self.get_tpf()
            else:
                tpf = self.tpf
            img = np.nanmedian(self.tpf.flux, axis=0)

            ax2 = axs[n, 1]
            ax = plot_aperture_outline(
                img, mask=self.aper_mask, imgwcs=tpf.wcs, ax=ax2
            )
            ax.axis("off")
        if return_lcs:
            return fig, custom_lcs
        else:
            return fig

    def get_flat_lc(
        self,
        lc,
        window_length=None,
        period=None,
        epoch=None,
        duration=None,
        method="biweight",
        return_trend=False,
        **wotan_kwargs,
    ):
        return _get_flat_lc(
            self=self,
            lc=lc,
            period=period,
            epoch=epoch,
            duration=duration,
            window_length=window_length,
            method=method,
            return_trend=return_trend,
            **wotan_kwargs,
        )

    def plot_trend_flat_lcs(
        self, lc, period=None, epoch=None, duration=None, binsize=10, **kwargs
    ):
        return _plot_trend_flat_lcs(
            self=self,
            lc=lc,
            period=period,
            epoch=epoch,
            duration=duration,
            binsize=binsize,
            **kwargs,
        )

    def plot_fold_lc(self, flat, period=None, epoch=None, ax=None):
        return _plot_fold_lc(
            self=self, flat=flat, period=period, epoch=epoch, ax=ax
        )

    def run_tls(self, flat, plot=True, **tls_kwargs):
        """
        """
        tls = transitleastsquares(t=flat.time, y=flat.flux, dy=flat.flux_err)
        tls_results = tls.power(**tls_kwargs)
        self.tls_results = tls_results
        if plot:
            fig = plot_tls(tls_results)
            fig.axes[0].set_title(f"{self.target_name} (sector {flat.sector})")
            return fig

    def plot_odd_even(
        self, flat, period=None, epoch=None, duration=None, ylim=None
    ):
        """
        """
        period = self.toi_period if period is None else period
        epoch = self.toi_epoch - TESS_TIME_OFFSET if epoch is None else epoch
        if (period is None) or (epoch is None):
            if self.tls_results is None:
                print("Running TLS")
                _ = self.run_tls(flat, plot=False)
            period = self.tls_results.period
            epoch = self.tls_results.T0
            ylim = self.tls_results.depth if ylim is None else ylim
        if ylim is None:
            ylim = 1 - self.toi_depth
        fig = plot_odd_even(
            flat, period=period, epoch=epoch, duration=duration, yline=ylim
        )
        fig.suptitle(f"{self.target_name} (sector {flat.sector})")
        return fig

    def get_transit_mask(self, lc, period=None, epoch=None, duration=None):
        """
        """
        period = self.toi_period if period is None else period
        epoch = self.toi_epoch - TESS_TIME_OFFSET if epoch is None else epoch
        duration = self.toi_duration if duration is None else duration
        tmask = get_transit_mask(
            lc, period=period, epoch=epoch, duration_hours=duration
        )
        return tmask

    @property
    def cadence(self):
        return "short"

    def save_lc(self, flat, fp=None):
        fold = flat.fold(
            period=self.toi_period, t0=self.toi_epoch - TESS_TIME_OFFSET
        )
        fold.time = fold.time * self.toi_period

        width = 3 * self.toi_duration / 24
        idx = (fold.time > -width) & (fold.time < width)
        if fp is None:
            fp = f"{self.target_name}_fold_lc.csv"
        fold[idx].remove_nans().to_csv(fp, index=False)
        print("Saved: ", fp)


"""
Functions below appear in both ShortCadence and LongCadence
Either class inherits different classes
"""


def _get_flat_lc(
    self,
    lc,
    period=None,
    epoch=None,
    duration=None,
    window_length=None,
    method="biweight",
    return_trend=False,
    **wotan_kwargs,
):
    """
    TODO: migrate self in class method;
    See plot_hrd in cluster.py
    """
    period = self.toi_period if period is None else period
    epoch = self.toi_epoch - TESS_TIME_OFFSET if epoch is None else epoch
    duration = self.toi_duration if duration is None else duration
    if duration is not None:
        if duration < 1:
            print("Duration should be in hours.")
    if window_length is None:
        window_length = 0.5 if duration is None else duration / 24 * 3
    if self.verbose:
        print(
            f"Using {method} filter with window_length={window_length:.2f} day"
        )
    if (period is not None) & (epoch is not None) & (duration is not None):
        tmask = get_transit_mask(
            lc, period=period, epoch=epoch, duration_hours=duration
        )
    else:
        tmask = np.zeros_like(lc.time, dtype=bool)
    # dummy holder
    flat, trend = lc.flatten(return_trend=True)
    # flatten using wotan
    wflat, wtrend = flatten(
        lc.time,
        lc.flux,
        method=method,
        mask=tmask,
        window_length=window_length,
        return_trend=True,
        **wotan_kwargs,
    )
    # overwrite
    flat.flux = wflat
    trend.flux = wtrend
    if return_trend:
        return flat, trend
    else:
        return flat


def _plot_trend_flat_lcs(
    self, lc, period=None, epoch=None, duration=None, binsize=10, **kwargs
):
    """
    plot trend and flat lightcurves (uses TOI ephemeris by default)

    TODO: migrate self in class method;
    See plot_hrd in cluster.py
    """
    period = self.toi_period if period is None else period
    epoch = self.toi_epoch - TESS_TIME_OFFSET if epoch is None else epoch
    duration = self.toi_duration if duration is None else duration
    if duration is not None:
        if duration < 1:
            print("Duration should be in hours.")

    fig, axs = pl.subplots(
        2, 1, figsize=(12, 10), constrained_layout=True, sharex=True
    )

    if (period is not None) & (epoch is not None) & (duration is not None):
        if self.verbose:
            print(
                f"Using period={period:.4f} d, epoch={epoch:.2f} BTJD, duration={duration:.2f} hr"
            )
        tmask = get_transit_mask(
            lc, period=period, epoch=epoch, duration_hours=duration
        )
    else:
        tmask = np.zeros_like(lc.time, dtype=bool)
    ax = axs.flatten()
    flat, trend = self.get_flat_lc(
        lc,
        period=period,
        epoch=epoch,
        duration=duration,
        return_trend=True,
        **kwargs,
    )
    lc[tmask].scatter(ax=ax[0], zorder=5, c="r", label="transit")
    if np.any(tmask):
        lc[~tmask].scatter(ax=ax[0], c="k", alpha=0.5, label="_nolegend_")
    ax[0].set_title(f"{self.target_name} (sector {lc.sector})")
    ax[0].set_xlabel("")
    trend.plot(ax=ax[0], c="b", lw=2, label="trend")

    if (period is not None) & (epoch is not None) & (duration is not None):
        tmask2 = get_transit_mask(
            flat, period=period, epoch=epoch, duration_hours=duration
        )
    else:
        tmask2 = np.zeros_like(lc.time, dtype=bool)
    flat.scatter(ax=ax[1], c="k", alpha=0.5, label="flat")
    if np.any(tmask2):
        flat[tmask2].scatter(ax=ax[1], zorder=5, c="r", s=10, label="transit")
    flat.bin(binsize).scatter(ax=ax[1], s=10, c="C1", label=f"bin ({binsize})")
    fig.subplots_adjust(hspace=0)
    return fig


def _plot_fold_lc(
    self, flat, period=None, epoch=None, duration=None, binsize=10, ax=None
):
    """
    plot folded lightcurve (uses TOI ephemeris by default)
    """
    if ax is None:
        fig, ax = pl.subplots(figsize=(12, 8))
    period = self.toi_period if period is None else period
    epoch = self.toi_epoch - TESS_TIME_OFFSET if epoch is None else epoch
    duration = self.toi_duration if duration is None else duration
    errmsg = "Provide period and epoch."
    assert (period is not None) & (epoch is not None), errmsg
    fold = flat.fold(period=period, t0=epoch)
    fold.scatter(ax=ax, c="k", alpha=0.5, label="raw")
    fold.bin(binsize).scatter(ax=ax, s=20, c="C1", label=f"bin {binsize}")
    if duration is None:
        if self.tls_results is not None:
            duration = self.tls_results.duration
    if duration is not None:
        xlim = 3 * duration / 24 / period
        ax.set_xlim(-xlim, xlim)
    ax.set_title(f"{self.target_name} (sector {flat.sector})")
    return ax


def _plot_pixel_lcs(self, mask=None):
    """
    Experimental: See eleanor.visualization.pixel_by_pixel():

    import eleanor

    target = eleanor.Source(tic=star.ticid, sector='recent', tc=True)
    data = eleanor.TargetData(target, height=5, width=5)
    vis = eleanor.Visualize(data)
    fig = vis.pixel_by_pixel(data_type="corrected", color_by_pixel=True)
    [ax.set_ylim(0.99,1.01) for n,ax in enumerate(fig.axes) if n>1];

    """
    mask = self.aper_mask if mask is None else mask
    y, x = np.nonzero(mask)

    nrows = max(x) - min(x) + 1
    ncols = max(y) - min(y) + 1
    fig, axs = pl.subplots(
        nrows,
        ncols,
        figsize=(10, 10),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0, "wspace": 0},
    )

    bkg = np.median(self.tpf.flux[:, ~mask], axis=1)
    for (i, j) in zip(x, y):
        flux = self.tpf.flux[:, j, i] - bkg
        flux /= np.nanmedian(flux)
        flux = linear_detrend(flux, bp=len(flux) // 2) + 1
        time = self.tpf.time
        ax = axs[j - min(y), i - min(x)]
        ax.plot(time, flux, label=f"(y,x)=({j},{i})")
        ax.legend(loc=3)
        ax.set_ylim(0, 1.3)
        if j != y[-1]:
            ax.set_xlabel("")
    return fig

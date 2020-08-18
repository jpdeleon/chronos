# -*- coding: utf-8 -*-

r"""
classes for k2 lightcurves produced by EVEREST and K2SFF pipelines
"""
# Import standard library
from pathlib import Path
from os.path import join, exists
from urllib.request import urlretrieve
import logging
import requests

# Import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import astropy.units as u
from scipy.ndimage import zoom
import lightkurve as lk
from astropy.io import fits
from astropy.table import Table
from wotan import flatten
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astroquery.skyview import SkyView
from astroplan import FixedTarget
from astroplan.plots import plot_finder_image
from transitleastsquares import transitleastsquares

# Import from package
from chronos.config import DATA_PATH
from chronos.target import Target
from chronos.constants import K2_TIME_OFFSET, Kepler_pix_scale
from chronos.plot import plot_tls, plot_odd_even
from chronos.utils import (
    detrend,
    get_all_campaigns,
    get_transit_mask,
    PadWithZeros,
)

pl.style.use("default")
log = logging.getLogger(__name__)

__all__ = ["K2", "Everest", "K2sff", "plot_k2_campaign_fov"]


class _KeplerLightCurve(lk.KeplerLightCurve):
    """augments parent class by adding convenience methods"""

    def detrend(self, polyorder=1, break_tolerance=None):
        lc = self.copy()
        half = lc.time.shape[0] // 2
        if half % 2 == 0:
            # add 1 if even
            half += 1
        return lc.flatten(
            window_length=half,
            polyorder=polyorder,
            break_tolerance=break_tolerance,
        )


class K2(Target):
    """
    sap and pdcsap
    """

    def __init__(
        self,
        epicid=None,
        campaign=None,
        gaiaDR2id=None,
        name=None,
        ra_deg=None,
        dec_deg=None,
        search_radius=3,
        quality_bitmask="default",
        verbose=True,
        clobber=True,
    ):
        super().__init__(
            epicid=epicid,
            gaiaDR2id=gaiaDR2id,
            name=name,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            search_radius=search_radius,
            verbose=verbose,
            clobber=clobber,
            mission="k2",
        )
        if self.epicid is not None:
            # epicid is initialized in Target if name has EPIC
            self.epicid = epicid
        self.quality_bitmask = quality_bitmask
        self.campaign = campaign
        self.tpf = None
        self.lc_raw = None
        self.lc_custom = None
        self.lcf = None
        self.all_campaigns = get_all_campaigns(self.epicid)
        if self.campaign is None:
            self.campaign = self.all_campaigns[0]
            print(f"Available campaigns: {self.all_campaigns}")
            print(f"Using campaign={self.campaign}.")
        self.tls_results = None
        # if self.verbose:
        #     print(f"Target: {name}")
        self.K2_star_params = None

    def get_tpf(self):
        """
        FIXME: refactor to tpf.py?
        """
        res = lk.search_targetpixelfile(
            f"EPIC {self.epicid}", campaign=self.campaign, mission="K2"
        )
        tpf = res.download()
        self.tpf = tpf
        return tpf

    def get_lc(self, lctype="pdcsap", campaign=None, quality_bitmask=None):
        """
        FIXME: refactor to lightcurve.py?
        """
        campaign = campaign if campaign is not None else self.campaign
        quality_bitmask = (
            quality_bitmask if quality_bitmask else self.quality_bitmask
        )
        if self.lcf is not None:
            # reload lcf if already in memory
            if self.lcf.campaign == campaign:
                lcf = self.lcf
            else:
                query_str = (
                    f"EPIC {self.epicid}" if self.epicid else self.target_coord
                )
                if self.verbose:
                    print(
                        f"Searching lightcurvefile for {query_str} (campaign {campaign})"
                    )
                q = lk.search_lightcurvefile(
                    query_str, campaign=campaign, mission="K2"
                )
                if len(q) == 0:
                    if self.verbose:
                        print(
                            f"Searching lightcurvefile for {self.target_coord.to_string()} (campaign {campaign})"
                        )
                    q = lk.search_lightcurvefile(
                        self.target_coord, campaign=campaign, mission="K2"
                    )
                assert q is not None, "Empty result. Check long cadence."
                if self.verbose:
                    print(f"Found {len(q)} lightcurves")
                if (campaign == "all") & (len(self.all_campaigns) > 1):
                    NotImplementedError
                    # lcf = q.download_all(quality_bitmask=quality_bitmask)
                else:
                    lcf = q.download(quality_bitmask=quality_bitmask)
                self.lcf = lcf
        else:
            query_str = (
                f"EPIC {self.epicid}" if self.epicid else self.target_coord
            )
            if self.verbose:
                print(
                    f"Searching lightcurvefile for {query_str} (campaign {campaign})"
                )
            q = lk.search_lightcurvefile(
                query_str, campaign=campaign, mission="K2"
            )
            if len(q) == 0:
                if self.verbose:
                    print(
                        f"Searching lightcurvefile for ra,dec=({self.target_coord.to_string()}) (campaign {campaign})"
                    )
                q = lk.search_lightcurvefile(
                    self.target_coord, campaign=campaign, mission="K2"
                )
            assert q is not None, "Empty result. Check long cadence."
            if self.verbose:
                print(f"Found {len(q)} lightcurves")
            if (campaign == "all") & (len(self.all_campaigns) > 1):
                NotImplementedError
                # lcf = q.download_all(quality_bitmask=quality_bitmask)
            else:
                lcf = q.download(quality_bitmask=quality_bitmask)
            self.lcf = lcf
        assert lcf is not None, "Empty result. Check long cadence."
        sap = lcf.SAP_FLUX
        pdcsap = lcf.PDCSAP_FLUX
        if isinstance(lcf, lk.LightCurveFileCollection):
            # merge multi-campaign into one lc
            if len(lcf) > 1:
                sap0 = sap[0].normalize()
                sap = [sap0.append(l.normalize()) for l in sap[1:]][0]
                pdcsap0 = pdcsap[0].normalize()
                pdcsap = [pdcsap0.append(l.normalize()) for l in pdcsap[1:]][0]
            else:
                raise ValueError(
                    f"Only campaign {lcf[0].campaign} (in {self.all_campaigns}) is available"
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
        sap_mask=None,
        aper_radius=None,
        percentile=None,
        threshold_sigma=None,
        use_pld=True,
    ):
        raise NotImplementedError
        # lc = tpf.to_lightcurve()

    def get_flat_lc(
        self,
        lc,
        period=None,
        epoch=None,
        duration=None,
        window_length=None,
        method="biweight",
        sigma_upper=None,
        sigma_lower=None,
        return_trend=False,
    ):
        """
        """
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
            window_length=window_length,
            mask=tmask,
            return_trend=True,
        )
        # overwrite
        flat.flux = wflat
        trend.flux = wtrend
        # clean lc
        sigma_upper = 5 if sigma_upper is None else sigma_upper
        sigma_lower = 10 if sigma_lower is None else sigma_lower
        flat = flat.remove_nans().remove_outliers(
            sigma_upper=sigma_upper, sigma_lower=sigma_lower
        )
        if return_trend:
            return flat, trend
        else:
            return flat

    def plot_trend_flat_lcs(
        self, lc, period, epoch, duration, binsize=10, **kwargs
    ):
        """
        plot trend and falt lightcurves (uses TOI ephemeris by default)
        """
        if duration < 1:
            print("Duration should be in hours.")
        assert (
            (period is not None) & (epoch is not None) & (duration is not None)
        )
        if self.verbose:
            print(
                f"Using period={period:.4f} d, epoch={epoch:.2f} BTJD, duration={duration:.2f} hr"
            )
        fig, axs = pl.subplots(
            2, 1, figsize=(12, 10), constrained_layout=True, sharex=True
        )

        if (period is not None) & (epoch is not None) & (duration is not None):
            tmask = get_transit_mask(
                lc, period=period, epoch=epoch, duration_hours=duration
            )
        else:
            tmask = np.zeros_like(lc.time, dtype=bool)
        ax = axs.flatten()
        flat, trend = self.get_flat_lc(
            lc, period=period, duration=duration, return_trend=True, **kwargs
        )
        lc[tmask].scatter(ax=ax[0], c="r", zorder=5, label="transit")
        if np.any(tmask):
            lc[~tmask].scatter(ax=ax[0], c="k", alpha=0.5, label="_nolegend_")
        ax[0].set_title(f"{self.target_name} (campaign {lc.campaign})")
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
            flat[tmask2].scatter(
                ax=ax[1], zorder=5, c="r", s=10, label="transit"
            )
        flat.bin(binsize).scatter(
            ax=ax[1], s=10, c="C1", label=f"bin ({binsize})"
        )
        fig.subplots_adjust(hspace=0)
        return fig

    def run_tls(self, flat, plot=True, **tls_kwargs):
        """
        """
        tls = transitleastsquares(t=flat.time, y=flat.flux, dy=flat.flux_err)
        tls_results = tls.power(**tls_kwargs)
        self.tls_results = tls_results
        if plot:
            fig = plot_tls(tls_results)
            fig.axes[0].set_title(
                f"{self.target_name} (campaign {flat.campaign})"
            )
            return fig

    def plot_fold_lc(
        self, flat, period, epoch, duration=None, binsize=10, ax=None
    ):
        """
        plot folded lightcurve (uses TOI ephemeris by default)
        """
        if ax is None:
            fig, ax = pl.subplots(figsize=(12, 8))
        errmsg = "Provide period and epoch."
        assert (period is not None) & (epoch is not None), errmsg
        fold = flat.fold(period=period, t0=epoch)
        fold.scatter(ax=ax, c="k", alpha=0.5, label="folded")
        fold.bin(binsize).scatter(
            ax=ax, s=20, c="C1", label=f"bin ({binsize})"
        )
        if duration is None:
            if self.tls_results is not None:
                duration = self.tls_results.duration
        if duration is not None:
            xlim = 3 * duration / period
            ax.set_xlim(-xlim, xlim)
        ax.set_title(f"{self.target_name} (campaign {flat.campaign})")
        return ax

    def plot_odd_even(self, flat, period, epoch, ylim=None):
        """
        """
        if (period is None) or (epoch is None):
            if self.tls_results is None:
                print("Running TLS")
                _ = self.run_tls(flat, plot=False)
            period = self.tls_results.period
            epoch = self.tls_results.T0
            ylim = self.tls_results.depth if ylim is None else ylim
        if ylim is None:
            ylim = 1 - self.toi_depth
        fig = plot_odd_even(flat, period=period, epoch=epoch, yline=ylim)
        fig.suptitle(f"{self.target_name} (campaign {flat.campaign})")
        return fig

    def get_transit_mask(self, lc, period, epoch, duration_hours):
        """
        """
        tmask = get_transit_mask(
            lc, period=period, epoch=epoch, duration_hours=duration_hours
        )
        return tmask


class Everest(K2):
    """
    everest pipeline
    """

    def __init__(
        self,
        epicid=None,
        campaign=None,
        gaiaDR2id=None,
        name=None,
        ra_deg=None,
        dec_deg=None,
        quality_bitmask="default",
        verbose=True,
        flux_type="flux",  # or fcor
    ):
        super().__init__(
            epicid=epicid,
            campaign=campaign,
            gaiaDR2id=gaiaDR2id,
            name=name,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            verbose=verbose,
        )
        """Initialize Everest

        Attributes
        ----------

        """
        self.flux_type = flux_type.upper()
        url, filename = self.get_everest_url_and_fn()
        self.url = url
        self.filename = filename
        self.quality = None
        self.cadenceno = None
        self.everest_recarray = None
        self.lc_everest = self.get_everest_lc()
        if self.lc_everest.campaign is None:
            self.lc_everest.campaign = self.campaign

    def get_everest_url_and_fn(self, campaign=None):
        """
        Note: uses pipeline version 2:

        fits url e.g.
        http://archive.stsci.edu/hlsps/everest/v2/c06/212400000/32685/hlsp_everest_k2_llc_212432685-c06_kepler_v2.0_lc.fits
        """
        if campaign is None:
            campaign = self.campaign

        id_str = "{0:09d}".format(self.epicid)
        url = "https://archive.stsci.edu/hlsps/everest/v2/"
        if campaign == 10:
            url += "c{0:02d}2/{1}00000/{2}/".format(
                campaign, id_str[:4], id_str[4:]
            )
            fn = "hlsp_everest_k2_llc_{0}-c{1:02d}2_kepler_v2.0_lc.fits".format(
                id_str, campaign
            )
        else:
            url += "c{0:02d}/{1}00000/{2}/".format(
                campaign, id_str[:4], id_str[4:]
            )
            fn = "hlsp_everest_k2_llc_{0}-c{1:02d}_kepler_v2.0_lc.fits".format(
                id_str, campaign
            )
        return url + fn, fn

    def get_everest_lc(
        self,
        campaign=None,
        flux_type=None,
        quality_bitmask=None,
        normalize=True,
        remove_nans=True,
    ):
        """
        see also https://archive.stsci.edu/k2/hlsp/everest/search.php

        flux_type : str
            'flux' = PLD-de-trended flux; 'fcor' = de-trended flux with CBV correction
        normalize : bool
            divide flux (and err) by its median
        quality: str
            option to choose which cadences will be masked
        return_mask : bool
            returns time, flux, (err,) mask
        """
        flux_type = self.flux_type if flux_type is None else flux_type.upper()
        if quality_bitmask is None:
            quality_bitmask = self.quality_bitmask
        if campaign is None:
            campaign = self.campaign

        if self.verbose:
            print(
                f"Querying EPIC {self.epicid} (campaign {campaign}) EVEREST light curve from MAST..."
            )
        try:
            url, fn = self.get_everest_url_and_fn(campaign)
            with fits.open(url) as hl:
                if self.verbose:
                    print(hl.info())
                recarray = hl[1].data
                self.everest_recarray = recarray
                cols = recarray.columns.names
                assert (
                    flux_type in cols
                ), f"flux_type={flux_type} not in {cols}"
                time = recarray["TIME"]
                flux = recarray[flux_type]
                err = recarray["FRAW_ERR"]
                self.quality = recarray["quality"]
                # self.cadenceno = recarray['CADN']
                # if flux_type=='FRAW':
                # if quality_bitmask!='none':
                #     #apply Kepler data quality flags based on bitmask on raw flux
                #     qf = KeplerQualityFlags()
                #     options = list(qf.OPTIONS.keys())
                #     assert quality_bitmask in options, f"quality_bitmask={quality_bitmask} not in {options}"
                #     bitmask = qf.OPTIONS[quality_bitmask]
                #     qmask = qf.create_quality_mask(self.quality, bitmask)
                #     time, flux, err = time[qmask], flux[qmask], err[qmask]
            # remove nans
            if remove_nans:
                idx = np.isfinite(time) & np.isfinite(flux)
                time, flux, err = time[idx], flux[idx], err[idx]
            if normalize:
                err /= np.median(flux)  # divide by median of raw flux
                flux /= np.median(flux)
            time += K2_TIME_OFFSET
            # hack
            lc = _KeplerLightCurve(
                time=time,
                flux=flux,
                flux_err=err,
                flux_unit=u.Unit("electron/second"),
                # FIXME: only day works when using lc.to_periodogram()
                time_format="jd",  # TIMEUNIT?
                time_scale="tdb",  # TIMESYS?
                centroid_col=None,
                centroid_row=None,
                quality=None,  # self.quality,
                quality_bitmask=None,  # self.quality_bitmask,
                cadenceno=self.cadenceno,
                targetid=self.epicid,
            )
            return lc
        except Exception as e:
            print(e)


class K2sff(K2):
    """
    """

    def __init__(
        self,
        epicid=None,
        campaign=None,
        gaiaDR2id=None,
        name=None,
        ra_deg=None,
        dec_deg=None,
        quality_bitmask="default",
        verbose=True,
        clobber=True,
        flux_type="flux",  # or fcor
    ):
        super().__init__(
            epicid=epicid,
            campaign=campaign,
            gaiaDR2id=gaiaDR2id,
            name=name,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            verbose=verbose,
            clobber=clobber,
        )
        """Initialize Everest

        Attributes
        ----------

        """
        self.flux_type = flux_type.upper()
        url, filename = self.get_k2sff_url_and_fn()
        self.url = url
        self.filename = filename
        self.quality = None
        self.cadenceno = None
        self.k2sff_best_aper_mask = None
        self.k2sff_header = None
        self.k2sff_recarray = None
        self.lc_k2sff = self.get_k2sff_lc()
        if self.lc_k2sff.campaign is None:
            self.lc_k2sff.campaign = self.campaign

    def get_k2sff_url_and_fn(self, campaign=None, filetype="fits"):
        """
        Note: uses pipeline version 2: https://ui.adsabs.harvard.edu/abs/2015ApJ...800...59V/abstract
        c.f. version 1:
        https://ui.adsabs.harvard.edu/abs/2014PASP..126..948V/abstract
        https://ui.adsabs.harvard.edu/abs/2014arXiv1412.1827V/abstract

        version 1 readme: https://archive.stsci.edu/hlsps/k2sff/hlsp_k2sff_k2_lightcurve_all_kepler_v1_readme.txt
        fits url e.g.
        https://archive.stsci.edu/hlsps/k2sff/c06/212400000/32685/hlsp_k2sff_k2_lightcurve_212432685-c06_kepler_v1_llc.fits
        or: allfiles.tar.gz
        or: -default-aper.txt
        """
        if campaign is None:
            campaign = self.campaign
        id_str = "{0:09d}".format(self.epicid)
        url = "https://archive.stsci.edu/hlsps/k2sff/"
        if self.campaign == 10:
            url += "c{0:02d}2/{1}00000/{2}/".format(
                campaign, id_str[:4], id_str[4:]
            )
            fn = "hlsp_k2sff_k2_lightcurve_{0}-c{1:02d}2_kepler_v1_llc.{2}".format(
                id_str, campaign, filetype
            )
        else:
            url += "c{0:02d}/{1}00000/{2}/".format(
                campaign, id_str[:4], id_str[4:]
            )
            fn = "hlsp_k2sff_k2_lightcurve_{0}-c{1:02d}_kepler_v1_llc.{2}".format(
                id_str, campaign, filetype
            )
        return url + fn, fn

    def get_k2sff_lc(
        self, campaign=None, flux_type="fcor", normalize=True, remove_nans=True
    ):
        """
        see also https://archive.stsci.edu/k2/hlsp/k2sff/search.php

        Note: 'flux_type'='fraw' is already normalized from the pipeline
        so setting normalize=True (dividing again by its mean) only
        produces difference of ~0.04 ppt.
        """
        flux_type = self.flux_type if flux_type is None else flux_type.upper()
        if campaign is None:
            campaign = self.campaign

        if self.verbose:
            print(
                f"Querying EPIC {self.epicid} (campaign {campaign}) K2SFF light curve from MAST..."
            )

        try:
            url, fn = self.get_k2sff_url_and_fn(campaign)
            with fits.open(url) as hdulist:
                # stacked_img = hdulist[24].data
                # hdr = hdulist[24].header
                prf_apertures = hdulist[23].data
                cir_apertures = hdulist[22].data
                # best lc using best aperture mask
                lc_hdr = hdulist[1].header
                self.k2sff_header = lc_hdr
                best_aper_mask_shape = lc_hdr["MASKTYPE"][:3].lower()
                best_aper_mask_idx = lc_hdr["MASKINDE"]
                if self.verbose:
                    print(hdulist.info())
                    print(
                        f"best aperture: {best_aper_mask_shape} (idx={best_aper_mask_idx})"
                    )
                if best_aper_mask_shape == "cir":
                    best_aper_mask = cir_apertures[best_aper_mask_idx]
                else:
                    best_aper_mask = prf_apertures[best_aper_mask_idx]
                # lightcurves
                self.k2sff_best_aper_mask = best_aper_mask
                recarray = hdulist[1].data
                self.k2sff_recarray = recarray
                cols = recarray.columns.names
                assert (
                    flux_type in cols
                ), f"flux_type={flux_type} not in {cols}"
                time = recarray["T"]
                flux = recarray[flux_type]
            if remove_nans:
                idx = np.isfinite(time) & np.isfinite(flux)
                time, flux = time[idx], flux[idx]
            if normalize:
                flux /= np.median(flux)
            time += K2_TIME_OFFSET
            # hack
            lc = _KeplerLightCurve(
                time=time,
                flux=flux,
                # flux_err=err,
                flux_unit=u.Unit("electron/second"),
                # FIXME: only day works when using lc.to_periodogram()
                time_format="jd",  # TIMEUNIT?
                time_scale="tdb",  # TIMESYS?
                centroid_col=None,
                centroid_row=None,
                quality=None,  # self.quality,
                quality_bitmask=None,  # self.quality_bitmask,
                cadenceno=self.cadenceno,
                targetid=self.epicid,
            )
            return lc
        except Exception as e:
            print(e)

    def plot_gaia_sources_on_survey(
        self,
        gaia_sources=None,
        fov_rad=None,
        depth=0.0,
        kmax=1.0,
        survey="DSS2 Red",
        ax=None,
        figsize=None,
    ):
        """Plot (superpose) Gaia sources on archival image

        Parameters
        ----------
        gaia_sources : pd.DataFrame
            gaia sources table
        fov_rad : astropy.unit
            FOV radius
        survey : str
            image survey; see from astroquery.skyview import SkyView;
            SkyView.list_surveys()
        ax : axis
            subplot axis
        Returns
        -------
        ax : axis
            subplot axis
        """
        if self.tpf is None:
            tpf = self.get_tpf()
        else:
            tpf = self.tpf
        ny, nx = tpf.flux.shape[1:]

        if fov_rad is None:
            diag = np.sqrt(nx ** 2 + ny ** 2)
            fov_rad = (diag * Kepler_pix_scale).to(u.arcsec)
        if gaia_sources is None:
            gaia_sources = self.query_gaia_dr2_catalog(radius=fov_rad.value)
        if self.gaiaid is None:
            # _ = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)
            target_gaiaid = gaia_sources.loc[0, "source_id"]
        else:
            target_gaiaid = self.gaiaid

        assert len(gaia_sources) > 1, "gaia_sources contains single entry"
        # make aperture mask
        maskhdr = tpf.hdu[2].header  # self.k2sff_header
        # make aperture mask outline
        contour = np.zeros((ny, nx))
        contour[np.where(self.k2sff_best_aper_mask)] = 1
        contour = np.lib.pad(contour, 1, PadWithZeros)
        highres = zoom(contour, 100, order=0, mode="nearest")
        extent = np.array([-1, nx, -1, ny])

        if self.verbose:
            print(
                f"Querying {survey} ({fov_rad:.2f} x {fov_rad:.2f}) archival image"
            )
        # -----------create figure---------------#
        if ax is None:
            # get img hdu for subplot projection
            results = SkyView.get_images(
                position=self.target_coord.icrs.to_string(),
                coordinates="icrs",
                survey=survey,
                radius=fov_rad,
                grid=False,
            )
            if len(results) > 0:
                hdu = results[0][0]
            else:
                errmsg = (
                    "SkyView returned empty result. Try a different survey."
                )
                raise ValueError(errmsg)
            # create figure with subplot projection
            fig = pl.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection=WCS(hdu.header))
        # plot survey img
        # FIXME: SkyView throws error when self.target_coord.distance=nan
        coord = SkyCoord(ra=self.target_coord.ra, dec=self.target_coord.dec)
        fixed_target = FixedTarget(coord, name=self.target_name)
        nax, hdu = plot_finder_image(
            fixed_target,
            ax=ax,
            fov_radius=fov_rad,
            survey=survey,
            reticle=False,
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

        for _, row in gaia_sources.iterrows():
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
        # orient such that north is up; left is east
        ax.invert_yaxis()
        if hasattr(ax, "coords"):
            ax.coords[0].set_major_formatter("dd:mm")
            ax.coords[1].set_major_formatter("dd:mm")
        # set img limits
        pl.setp(
            nax,
            xlim=(0, mx),
            ylim=(0, my),
            title=self.target_name
            # title="{0} ({1:.2f}' x {1:.2f}')".format(survey, fov_rad.value),
        )
        return ax

    def get_K2_star_params(self):
        """
        """
        if self.K2_star_params is None:
            fp = Path(DATA_PATH, "apjsab7230t1_mrt.txt")
            if not fp.exists():
                get_K2_star_data()
            tab = Table.read(fp, format="ascii")
            df = tab.to_pandas()
            self.K2_star_params = df
        else:
            df = self.K2_star_params
        q = df.query(f"EPIC=={self.epicid}")
        if len(q) > 1:
            return q.iloc[0]
        else:
            return q

    @property
    def K2_Rstar(self):
        q = self.get_K2_star_params()
        return q.Rstar

    @property
    def K2_Rstar_errs(self):
        q = self.get_K2_star_params()
        return q[["e_Rstar", "E_Rstar"]]

    @property
    def K2_Mstar(self):
        q = self.get_K2_star_params()
        return q.Mstar

    @property
    def K2_Mstar_errs(self):
        q = self.get_K2_star_params()
        return q[["e_Mstar", "E_Mstar"]]


def get_K2_star_data():
    """
    Hardegree-Ullmann+2020:
    https://iopscience.iop.org/0067-0049/247/1/28/suppdata/apjsab7230t1_mrt.txt
    """
    url = "https://iopscience.iop.org/0067-0049/247/1/28/suppdata/apjsab7230t1_mrt.txt"
    fp = Path(DATA_PATH, "apjsab7230t1_mrt.txt")
    print("Downloading Table X of Hardegree-Ullmann+2020:")
    urlretrieve(url, fp)
    print("Saved: ", fp)


def get_K2_campaign_fov():
    """
    K2 campaign FOV footprint in json format
    Source:
    https://keplerscience.arc.nasa.gov/k2-fields.html
    """
    url = "https://raw.githubusercontent.com/KeplerGO/K2FootprintFiles/master/json/k2-footprint.json"
    footprint_dict = requests.get(url).json()
    return footprint_dict


def plot_k2_campaign_fov(
    campaign, figsize=(8, 5), ax=None, color="k", text_offset=0
):
    """
    plot FOV of a given K2 campaign
    """
    footprint_dict = get_K2_campaign_fov()
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=figsize)
    channels = footprint_dict[f"c{campaign}"]["channels"]

    for c in channels.keys():
        channel = channels[c]
        x = channel["corners_ra"] + channel["corners_ra"][:1]
        y = channel["corners_dec"] + channel["corners_dec"][:1]
        ax.plot(x, y, color=color)
    ax.annotate(
        campaign,
        (x[0] + text_offset, y[0] + text_offset),
        color=color,
        fontsize=20,
    )
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("DEC [deg]")
    return ax

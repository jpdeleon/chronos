# -*- coding: utf-8 -*-

r"""
Multi-Sector Light Curves From TESS Full Frame Images (DIAMANTE):
http://archive.stsci.edu/hlsp/diamante
"""

# Import standard library
from pathlib import Path
import logging

# Import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import astropy.units as u
from astropy.io import fits
from wotan import flatten
from transitleastsquares import transitleastsquares

# Import from package
from chronos.config import DATA_PATH
from chronos.target import Target
from chronos.plot import plot_tls, plot_odd_even
from chronos.tpf import FFI_cutout
from chronos.constants import TESS_TIME_OFFSET
from chronos.utils import (
    get_tois,
    get_transit_mask,
    parse_aperture_mask,
    TessLightCurve,
)

log = logging.getLogger(__name__)

__all__ = ["Diamante"]


class Diamante(Target):
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
        quality_bitmask=None,
        search_radius=3,
        # mission="tess",
        aper_radius=2,
        lc_num=1,
        verbose=True,
        clobber=True,
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
            verbose=verbose,
        )
        """Initialize Diamante.
        See http://archive.stsci.edu/hlsp/diamante

        Attributes
        ----------
        lc_num : int
            [1,2]
            1 : co-trended lc; 2: normalized lc
        aper_radius : int
            [1,2] pix (default=2)
        """
        self.base_url = "https://archive.stsci.edu/hlsps/diamante"
        self.diamante_catalog = self.get_diamante_catalog()
        self.new_diamante_candidates = self.get_new_diamante_candidates()
        self.candidate_params = self.get_candidate_ephemeris()
        self.sectors = self.all_sectors  # multi-sectors
        self.sector = self.sectors[0]

        if self.gaiaid is None:
            _ = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)

        self.lc_num = lc_num
        self.lc_nums = [1, 2]
        if self.lc_num not in self.lc_nums:
            raise ValueError(f"Type not among {self.lc_nums}")
        self.aper_radius = aper_radius
        self.apers = [1, 2]
        if self.aper_radius not in self.apers:
            raise ValueError(f"Type not among {self.apers}")
        self.quality_bitmask = quality_bitmask
        self.fits_url = None
        self.hdulist = None
        self.header0 = None
        self.data, self.header = self.get_diamante_fits()
        self.lc = self.get_diamante_lc()
        self.lc.targetid = self.ticid
        self.time = self.lc.time
        self.flux = self.lc.flux
        self.err = self.lc.flux_err
        self.sap_mask = "round"
        # self.threshold_sigma = 5  # dummy
        # self.percentile = 95  # dummy
        self.cutout_size = (15, 15)  # dummy
        self.tpf_tesscut = None
        self.ffi_cutout = None
        self.aper_mask = None
        self.contratio = None

    def get_diamante_catalog(self):
        """
        """
        diamante_catalog_fp = Path(DATA_PATH, "diamante_catalog.csv")
        if diamante_catalog_fp.exists():
            df = pd.read_csv(diamante_catalog_fp)
        else:
            url = f"{self.base_url}/hlsp_diamante_tess_lightcurve_catalog_tess_v1_cat.csv"
            df = pd.read_csv(url)
            df.to_csv(diamante_catalog_fp, index=False)
        return df

    def get_new_diamante_candidates(self):
        """
        """
        tois = get_tois()
        df = self.diamante_catalog.copy()
        idx = df["#ticID"].isin(tois["TIC ID"])
        return df[~idx]

    def get_diamante_url(self, ext="fits"):
        """
        hlsp_diamante_tess_lightcurve_tic-<id>_tess_v1_<ext>
        where:

        <id> = the full, zero-padded, 16-digit TIC ID
        <ext> = type of file product, one of "llc.fits", "llc.txt", or "dv.pdf"

        https://archive.stsci.edu/hlsps/diamante/0000/0009/0167/4675/
        hlsp_diamante_tess_lightcurve_tic-0000000901674675_tess_v1_llc.fits
        """
        if not np.any(self.diamante_catalog["#ticID"].isin([self.ticid])):
            raise ValueError(f"TIC {self.ticid} not in DIAmante catalog.")
        tid = f"{self.ticid}".zfill(16)
        dir = f"{tid[0:4]}/{tid[4:8]}/{tid[8:12]}/{tid[12:16]}"
        fp = f"{self.base_url}/{dir}/hlsp_diamante_tess_lightcurve_tic-{tid}_tess_v1_llc.{ext}"
        return fp

    def get_diamante_fits(self):
        """get target and light curve header and data
        """
        fp = self.get_diamante_url()
        try:
            hdulist = fits.open(fp)
            if self.verbose:
                print(hdulist.info())
            lc_data = hdulist[1].data
            lc_header = hdulist[1].header

            # set
            self.fits_url = fp
            self.hdulist = hdulist
            self.header0 = hdulist[0].header
            return lc_data, lc_header

        except Exception:
            msg = f"File not found:\n{fp}\n"
            raise ValueError(msg)

    def get_diamante_lc(self, lc_num=None, aper_radius=None, sort=True):
        """
        Parameters
        ----------
        lc_type : int

        """
        aper_radius = self.aper_radius if aper_radius is None else aper_radius
        lc_num = self.lc_num if lc_num is None else lc_num
        assert lc_num in self.lc_nums
        assert aper_radius in self.apers

        if self.verbose:
            print(f"Using DIAmante LC{lc_num} (rad={aper_radius}) lightcurve.")

        time = self.data["BTJD"] + 2457000  # BJD, days
        flux = self.data[f"LC{lc_num}_AP{aper_radius}"]
        err = self.data[f"ELC{lc_num}_AP{aper_radius}"]
        quality = self.data[f"FLAG_AP{lc_num}"]
        if sort:
            idx = np.argsort(time)
        else:
            idx = np.ones_like(time, bool)
        # hack tess lightkurve
        return TessLightCurve(
            time=time[idx],
            flux=flux[idx],
            flux_err=err[idx],
            # FIXME: only day works when using lc.to_periodogram()
            time_format="jd",  # TIMEUNIT is d in fits header
            time_scale="tdb",  # TIMESYS in fits header
            # centroid_col=None,
            # centroid_row=None,
            quality=quality,
            quality_bitmask=self.quality_bitmask,
            # cadenceno=cadence,
            sector=self.sectors,
            targetid=self.ticid,
            ra=self.target_coord.ra.deg,
            dec=self.target_coord.dec.deg,
            label=None,
            meta=None,
        ).normalize()

    def plot_all_lcs(self, sigma=10):
        """
        """
        # lcs = {}
        fig, ax = pl.subplots(1, 1, figsize=(10, 6))
        for aper in [1, 2]:
            lc = self.get_diamante_lc(
                lc_num=1, aper_radius=aper
            ).remove_outliers(sigma=sigma)
            lc.scatter(ax=ax, label=f"aper={aper}")
            # lcs[aper] = lc
        ax.set_title(f"{self.target_name} (sector {self.sector})")
        ax.legend()
        return fig

    def validate_target_header(self):
        """
        see self.header0
        """
        assert self.header0["OBJECT"] == self.target_name
        raise NotImplementedError()

    def get_aper_mask_diamante(self, sap_mask="round"):
        """
        This is an estimate of DIAmante aperture based on aper
        """
        print(f"Estimating DIAmante aperture using r={self.aper_radius} pix.")
        if self.ffi_cutout is None:
            # first download tpf cutout
            self.ffi_cutout = FFI_cutout(
                sector=self.sector,
                gaiaDR2id=self.gaiaid,
                toiid=self.toiid,
                ticid=self.ticid,
                search_radius=self.search_radius,
                quality_bitmask=self.quality_bitmask,
            )
        self.tpf_tesscut = self.ffi_cutout.get_tpf_tesscut()
        aper_mask = parse_aperture_mask(
            self.tpf_tesscut, sap_mask=sap_mask, aper_radius=self.aper_radius
        )
        self.aper_mask = aper_mask
        return aper_mask

    def get_candidate_ephemeris(self):
        df = self.new_diamante_candidates
        d = df[df["#ticID"] == self.ticid].squeeze()
        self.period = d["periodBLS"]
        self.epoch = d["t0Fit"] + TESS_TIME_OFFSET
        self.duration = d["duration"]
        self.depth = d["trdepth"] * 1e-6
        return d.copy()

    def get_flat_lc(
        self,
        lc=None,
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
        lc = self.lc if lc is None else lc
        period = self.period if period is None else period
        epoch = self.epoch if epoch is None else epoch
        duration = self.duration if duration is None else duration
        duration_hours = duration * 24
        if duration_hours < 1:
            raise ValueError("Duration should be in hours.")
        if window_length is None:
            window_length = 0.5 if duration is None else duration * 3
        if self.verbose:
            print(
                f"Using {method} filter with window_length={window_length:.2f} day"
            )
        if (period is not None) & (epoch is not None) & (duration is not None):
            tmask = get_transit_mask(
                lc, period=period, epoch=epoch, duration_hours=duration_hours
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
        flat = (
            flat.remove_nans()
        )  # .remove_outliers(sigma_upper=sigma_upper, sigma_lower=sigma_lower)
        if return_trend:
            return flat, trend
        else:
            return flat

    def plot_trend_flat_lcs(
        self,
        lc=None,
        period=None,
        epoch=None,
        duration=None,
        binsize=10,
        **kwargs,
    ):
        """
        plot trend and falt lightcurves (uses TOI ephemeris by default)
        """
        lc = self.lc if lc is None else lc
        period = self.period if period is None else period
        epoch = self.epoch if epoch is None else epoch
        duration = self.duration if duration is None else duration
        duration_hours = duration * 24
        if duration_hours < 1:
            raise ValueError("Duration should be in hours.")
        assert (
            (period is not None) & (epoch is not None) & (duration is not None)
        )
        if self.verbose:
            print(
                f"Using period={period:.4f} d, epoch={epoch:.2f} BTJD, duration={duration_hours:.2f} hr"
            )
        fig, axs = pl.subplots(
            2, 1, figsize=(12, 10), constrained_layout=True, sharex=True
        )

        if (period is not None) & (epoch is not None) & (duration is not None):
            tmask = get_transit_mask(
                lc, period=period, epoch=epoch, duration_hours=duration_hours
            )
        else:
            tmask = np.zeros_like(lc.time, dtype=bool)
        ax = axs.flatten()
        flat, trend = self.get_flat_lc(
            lc,
            period=period,
            duration=duration_hours,
            return_trend=True,
            **kwargs,
        )
        lc[tmask].scatter(ax=ax[0], c="r", zorder=5, label="transit")
        if np.any(tmask):
            lc[~tmask].scatter(ax=ax[0], c="k", alpha=0.5, label="_nolegend_")
        ax[0].set_title(f"{self.target_name} (sector {lc.sector})")
        ax[0].set_xlabel("")
        trend.plot(ax=ax[0], c="b", lw=2, label="trend")

        if (period is not None) & (epoch is not None) & (duration is not None):
            tmask2 = get_transit_mask(
                flat, period=period, epoch=epoch, duration_hours=duration_hours
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
            fig.axes[0].set_title(f"{self.target_name} (sector {flat.sector})")
            return fig

    def plot_fold_lc(
        self, flat, period=None, epoch=None, duration=None, binsize=10, ax=None
    ):
        """
        plot folded lightcurve (uses TOI ephemeris by default)
        """
        period = self.period if period is None else period
        epoch = self.epoch if epoch is None else epoch
        duration = self.duration if duration is None else duration
        if duration is None:
            if self.tls_results is not None:
                duration = self.tls_results.duration
        duration_hours = duration * 24
        if duration_hours < 1:
            raise ValueError("Duration should be in hours.")
        if ax is None:
            fig, ax = pl.subplots(figsize=(12, 8))
        errmsg = "Provide period and epoch."
        assert (period is not None) & (epoch is not None), errmsg
        fold = flat.fold(period=period, t0=epoch)
        fold.scatter(ax=ax, c="k", alpha=0.5, label="folded")
        fold.bin(binsize).scatter(
            ax=ax, s=20, c="C1", label=f"bin ({binsize})"
        )
        if duration is not None:
            xlim = 3 * duration / period
            ax.set_xlim(-xlim, xlim)
        ax.set_title(f"{self.target_name} (sector {flat.sector})")
        return ax

    def plot_odd_even(
        self, flat, period=None, epoch=None, duration=None, ylim=None
    ):
        """
        """
        period = self.period if period is None else period
        epoch = self.epoch - TESS_TIME_OFFSET if epoch is None else epoch
        duration = self.duration if duration is None else duration
        if (period is None) or (epoch is None):
            if self.tls_results is None:
                print("Running TLS")
                _ = self.run_tls(flat, plot=False)
            period = self.tls_results.period
            epoch = self.tls_results.T0
            ylim = self.tls_results.depth if ylim is None else ylim
        if ylim is None:
            ylim = 1 - self.depth
        fig = plot_odd_even(
            flat, period=period, epoch=epoch, duration=duration, yline=ylim
        )
        fig.suptitle(f"{self.target_name} (sector {flat.sector})")
        return fig

    def get_transit_mask(self, lc, period, epoch, duration_hours):
        """
        """
        period = self.period if period is None else period
        epoch = self.epoch if epoch is None else epoch
        duration_hours = (
            self.duration * 24 if duration_hours is None else duration_hours
        )
        if duration_hours < 1:
            raise ValueError("Duration should be in hours.")
        tmask = get_transit_mask(
            lc, period=period, epoch=epoch, duration_hours=duration_hours
        )
        return tmask

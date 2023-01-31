# -*- coding: utf-8 -*-

r"""
classes for working with lightcurves from the PATHOS pipeline:
http://archive.stsci.edu/hlsp/pathos
"""

# Import standard library
from pathlib import Path
import logging

# Import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from astroquery.mast import Observations
from astropy.io import fits
from wotan import flatten
from transitleastsquares import transitleastsquares

# Import from package
from chronos.config import DATA_PATH
from chronos.target import Target

from chronos.tpf import FFI_cutout
from chronos.plot import plot_tls, plot_odd_even
from chronos.utils import get_transit_mask, parse_aperture_mask, TessLightCurve
from chronos.constants import TESS_TIME_OFFSET

PATHOS_SECTORS = np.arange(1, 27, 1)
PATHOS_PAPER = "https://ui.adsabs.harvard.edu/abs/2020arXiv200512281N/abstract"
PATHOS_README = "https://archive.stsci.edu/hlsps/pathos/hlsp_pathos_tess_lightcurve_all_tess_v1_readme.txt"

log = logging.getLogger(__name__)

__all__ = ["PATHOS"]


class PATHOS(Target):
    def __init__(
        self,
        sector=None,
        cam=None,
        ccd=None,
        name=None,
        toiid=None,
        ticid=None,
        epicid=None,
        gaiaDR2id=None,
        ra_deg=None,
        dec_deg=None,
        quality_bitmask=None,
        search_radius=3,
        lctype="corr",
        aper_idx=4,
        mission="tess",
        verbose=True,
        clobber=False,
        # mission=("Kepler", "K2", "TESS"),
        # quarter=None,
        # month=None,
        # campaign=None,
        # limit=None,
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
        """Initialize PATHOS.
        See http://archive.stsci.edu/hlsp/pathos

        Attributes
        ----------
        aper_idx : int
            PATHOS aperture index: [1,2,3,4] pix in radius
        lctype: str
            PATHOS lc types: ["raw", "corr"]
        """
        if self.verbose:
            print("Using PATHOS lightcurve.")
        self.sector = sector
        if self.sector is None:
            print(f"Available sectors: {self.all_sectors}")
            if len(self.all_sectors) != 1:
                idx = [
                    True if s in PATHOS_SECTORS else False
                    for s in self.all_sectors
                ]
                if sum(idx) == 0:
                    msg = f"PATHOS lc is currently available for sectors={PATHOS_SECTORS}\n"
                    raise ValueError(msg)
                if sum(idx) == 1:
                    self.sector = self.all_sectors[idx][
                        0
                    ]  # get first available
                else:
                    self.sector = self.all_sectors[idx][
                        0
                    ]  # get first available
                    # get first available
                    print(
                        f"PATHOS lc may be available for sectors {self.all_sectors[idx]}"
                    )
            print(f"Using sector={self.sector}.")
        self.mast_table = self.get_mast_table()
        if self.gaiaid is None:
            _ = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)
        self.lctype = lctype
        self.lctypes = ["raw", "corr"]
        self.aper_idx = str(aper_idx)
        assert self.aper_idx in [
            "1",
            "2",
            "3",
            "4",
        ], "PATHOS has only [1,2,3,4] pix aperture radius"
        self.fits_url = None
        self.header0 = None  # target header
        self.hdulist = None
        self.data, self.header = self.get_pathos_fits()
        self.quality_bitmask = quality_bitmask
        self.lc = self.get_pathos_lc()
        self.pathos_candidates = self.get_pathos_candidates()
        self.tpf_tesscut = None
        self.ffi_cutout = None
        self.aper_mask = None

    def get_pathos_candidates(self):
        fp = Path(DATA_PATH, "pathos_candidates.csv")
        if (not fp.exists()) or self.clobber:
            d = pd.read_html("http://archive.stsci.edu/hlsp/pathos")[0]
            d.to_csv(fp, index=False)
            if self.clobber:
                print("Saved: ", fp)
        else:
            d = pd.read_csv(fp)
            if self.clobber:
                print("Loaded: ", fp)
        return d

    def get_mast_table(self):
        """https://archive.stsci.edu/hlsp/cdips"""
        if self.gaia_params is None:
            _ = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)
        if self.tic_params is None:
            _ = self.query_tic_catalog(return_nearest_xmatch=True)
        if not self.validate_gaia_tic_xmatch():
            raise ValueError("Gaia and Tic Catalog match failed")
        mast_table = Observations.query_criteria(
            target_name=self.ticid, provenance_name="PATHOS"
        )
        if len(mast_table) == 0:
            raise ValueError("No PATHOS lightcurve in MAST.")
        else:
            print(f"Found {len(mast_table)} PATHOS lightcurves.")
        return mast_table.to_pandas()

    def get_pathos_url(self):
        """
        Each target has a FITS and TXT version of the light curves available.
        The files are stored in sub-directories based on the Sector they are
        in as a 4-digit, zero-padded number, e.g., "s0001/" for Sector 1.
        The data file naming convention is:

        hlsp_pathos_tess_lightcurve_tic-<ticid>-<sector>_tess_v1_<ext>
        """
        base = "https://archive.stsci.edu/hlsps/pathos/"
        assert self.sector is not None
        assert self.gaiaid is not None
        tic = str(self.ticid).zfill(10)
        sect = str(self.sector).zfill(4)
        url = (
            base
            + f"s{sect}/hlsp_pathos_tess_lightcurve_tic-{tic}-s{sect}_tess_v1_llc.fits"
        )
        return url

    def get_pathos_fits(self):
        """get pathos target and light curve header and data"""
        fp = self.get_pathos_url()
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
            # msg += f"Using sector={self.sector} in {self.all_sectors}.\n"
            raise ValueError(msg)

    def get_pathos_lc(self, lctype=None, aper_idx=None, sort=True):
        """
        Parameters
        ----------
        """
        aper = aper_idx if aper_idx is not None else self.aper_idx
        lctype = lctype if lctype is not None else self.lctype

        tstr = "TIME"
        if lctype == "raw":
            fstr = f"AP{aper}_FLUX_RAW"
        elif lctype == "corr":
            # tstr = "TIMECORR"
            fstr = f"AP{aper}_FLUX_COR"
        else:
            raise ValueError(" or ".join(self.lctypes))
        # barycentric-corrected, truncated TESS Julian Date (BJD - 2457000.0)
        time = self.data[tstr]
        flux = self.data[fstr]
        # err = self.data[estr]
        xpos = self.data["X_POSITION"]
        ypos = self.data["Y_POSITION"]
        if sort:
            idx = np.argsort(time)
        else:
            idx = np.ones_like(time, bool)
        # hack tess lightkurve
        return TessLightCurve(
            time=time[idx],
            flux=flux[idx],
            # flux_err=err[idx],
            # FIXME: only day works when using lc.to_periodogram()
            time_format="jd",  # TIMEUNIT is bjd in fits header
            time_scale="tdb",  # TIMESYS in fits header
            centroid_col=ypos,
            centroid_row=xpos,
            quality=None,
            quality_bitmask=self.quality_bitmask,
            cadenceno=None,
            sector=self.sector,
            camera=self.header0["CAMERA"],
            ccd=self.header0["CCD"],
            targetid=self.toi_params["TIC ID"]
            if self.toi_params is not None
            else self.ticid,
            ra=self.target_coord.ra.deg,
            dec=self.target_coord.dec.deg,
            label=None,
            meta=None,
        ).normalize()

    def get_aper_mask_pathos(self, sap_mask="round"):
        """
        This is an estimate of PATHOS aperture only
        """
        print(
            f"PATHOS has no aperture info in fits. Estimating aperture instead using aper_idx={self.aper_idx} pix."
        )
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
        idx = int(self.aper_idx) - 1  #
        aper_mask = parse_aperture_mask(
            self.tpf_tesscut, sap_mask=sap_mask, aper_radius=idx
        )
        self.aper_mask = aper_mask
        return aper_mask

    def validate_target_header(self):
        """
        see self.header0['sector']==self.sector
        """
        raise NotImplementedError()

    def plot_all_lcs(self, lctype="corr", sigma=10):
        """ """
        pathos_lcs = {}
        fig, ax = pl.subplots(1, 1, figsize=(10, 6))
        for aper in [1, 2, 3, 4]:
            lc = self.get_pathos_lc(
                lctype=lctype, aper_idx=aper
            ).remove_outliers(sigma=sigma)
            lc.scatter(ax=ax, label=f"aper={aper}")
            pathos_lcs[aper] = lc
        ax.set_title(f"{self.target_name} (sector {self.sector})")
        ax.legend(title=f"lc={lctype}")
        return fig

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
        """ """
        if duration < 1:
            raise ValueError("Duration should be in hours.")
        if window_length is None:
            window_length = 0.5 if duration is None else duration / 24 * 3
        if self.verbose:
            print(
                f"Using {method} filter with window_length={window_length:.2f} day"
            )
        if (period is not None) & (epoch is not None) & (duration is not None):
            tmask = get_transit_mask(
                lc.time, period=period, t0=epoch, dur=duration / 24
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
            raise ValueError("Duration should be in hours.")
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
                lc.time, period=period, t0=epoch, dur=duration / 24
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
            flat[tmask2].scatter(
                ax=ax[1], zorder=5, c="r", s=10, label="transit"
            )
        flat.bin(binsize).scatter(
            ax=ax[1], s=10, c="C1", label=f"bin ({binsize})"
        )
        fig.subplots_adjust(hspace=0)
        return fig

    def run_tls(self, flat, plot=True, **tls_kwargs):
        """ """
        tls = transitleastsquares(t=flat.time, y=flat.flux, dy=flat.flux_err)
        tls_results = tls.power(**tls_kwargs)
        self.tls_results = tls_results
        if plot:
            fig = plot_tls(tls_results)
            fig.axes[0].set_title(f"{self.target_name} (sector {flat.sector})")
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
        ax.set_title(f"{self.target_name} (sector {flat.sector})")
        return ax

    def plot_odd_even(self, flat, period=None, epoch=None, ylim=None):
        """ """
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
        fig = plot_odd_even(flat, period=period, epoch=epoch, yline=ylim)
        fig.suptitle(f"{self.target_name} (sector {flat.sector})")
        return fig

    def get_transit_mask(self, lc, period, epoch, duration_hours):
        """ """
        tmask = get_transit_mask(
            lc.time, period=period, t0=epoch, dur=duration_hours / 24
        )
        return tmask

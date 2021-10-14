# -*- coding: utf-8 -*-

r"""
classes for working with lightcurves from the CDIPS pipeline
"""
# Import standard library
from os.path import join, exists
import logging

# Import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import astropy.units as u
from astroquery.mast import Observations
from astropy.io import fits
from wotan import flatten
from transitleastsquares import transitleastsquares

# Import from package
from chronos.config import DATA_PATH
from chronos.target import Target
from chronos.tpf import FFI_cutout
from chronos.plot import plot_tls, plot_odd_even
from chronos.utils import (
    get_ctois,
    get_sector_cam_ccd,
    parse_aperture_mask,
    get_transit_mask,
    TessLightCurve,
)
from chronos.constants import TESS_TIME_OFFSET

log = logging.getLogger(__name__)

__all__ = ["CDIPS", "get_cdips_inventory", "get_url_in_cdips_inventory"]

CDIPS_SECTORS = np.arange(1, 14, 1)
CDIPS_APER_PIX = [1, 1.5, 2.25]
CDIPS_PAPER = "https://ui.adsabs.harvard.edu/abs/2019ApJS..245...13B/abstract"
CDIPS_REPORT = "http://lgbouma.com/cdips_documentation/20191127_vetting_report_description_document.pdf"
CDIPS_MAST_README = "https://archive.stsci.edu/hlsps/cdips/hlsp_cdips_tess_ffi_all_tess_v01_readme.md"
CDIPS_PIPELINE_CODE = "https://github.com/waqasbhatti/cdips-pipeline"
CDIPS_CODE = "https://github.com/lgbouma/cdips"
CDIPS_CANDIDATES = "https://github.com/lgbouma/cdips_followup/blob/master/data/candidate_database/candidates.csv"


class CDIPS(Target):
    """
    The primary header contains information about the target star, including the
    catalogs that claimed cluster membership or youth (`CDIPSREF`), and a key that
    enables back-referencing to those catalogs in order to discover whatever those
    investigators said about the object (`CDEXTCAT`). Membership claims based on
    Gaia-DR2 data are typically the highest quality claims. Cross-matches against
    TICv8 and Gaia-DR2 are also included.
    """

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
        lctype="flux",
        aper_idx=1,
        mission="tess",
        verbose=True,
        clobber=True,
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
        """Initialize CDIPS

        Attributes
        ----------
        aper_idx : str
            CDIPS aperture index: [1,2,3] which is [1,1.5,2.25] pix in radius
        lctype: str
            CDIPS lc types: ["flux", "mag", "tfa", "pca"]
        """
        if self.verbose:
            print("Using CDIPS lightcurve.")
        self.sector = sector
        if self.sector is None:
            print(f"Available sectors: {self.all_sectors}")
            if len(self.all_sectors) != 1:
                idx = [
                    True if s in CDIPS_SECTORS else False
                    for s in self.all_sectors
                ]
                if sum(idx) == 0:
                    msg = f"CDIPS lc is currently available for sectors={CDIPS_SECTORS}\n"
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
                        f"CDIPS lc may be available for sectors {self.all_sectors[idx]}"
                    )
            print(f"Using sector={self.sector}.")
        self.mast_table = self.get_mast_table()
        self.cam = cam
        self.ccd = ccd
        if (self.sector is None) | (self.cam is None) | (self.ccd is None):
            # overwrite
            sector0, cam0, ccd0 = get_sector_cam_ccd(
                self.target_coord, self.sector
            )
            self.cam = cam0
            self.ccd = ccd0
        else:
            assert self.cam == cam0
            assert self.ccd == ccd

        if self.gaiaid is None:
            _ = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)

        # self.mission = mission
        self.lctype = lctype
        self.lctypes = ["flux", "mag", "tfa", "pca"]
        if self.lctype not in self.lctypes:
            raise ValueError(f"Type not among {self.lctypes}")
        self.aper_idx = str(aper_idx)
        assert self.aper_idx in [
            "1",
            "2",
            "3",
        ], "CDIPS has only [1,2,3] aperture indices"
        self.quality_bitmask = quality_bitmask
        self.fits_url = None
        self.header0 = None  # target header
        self.catalog_ref = None  # references
        self.catalog_gaiaids = None  # gaia id(s) in catalog_ref
        self.hdulist = None
        # self.ccd_info = Tesscut.get_sectors(self.target_coord).to_pandas()
        self.data, self.header = self.get_cdips_fits()
        self.lc = self.get_cdips_lc()
        self.lc.targetid = self.ticid
        self.cadence = self.header["XPOSURE"] * u.second  # .to(u.minute)
        self.time = self.lc.time
        self.flux = self.lc.flux.value
        self.err = self.lc.flux_err.value
        ctois = get_ctois()
        self.cdips_candidates = ctois[ctois["User"] == "bouma"]
        self.tpf_tesscut = None
        self.ffi_cutout = None
        self.aper_mask = None

    def get_mast_table(self):
        """https://archive.stsci.edu/hlsp/cdips
        """
        if self.gaia_params is None:
            _ = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)
        if self.tic_params is None:
            _ = self.query_tic_catalog(return_nearest_xmatch=True)
        if not self.validate_gaia_tic_xmatch():
            raise ValueError("Gaia and Tic Catalog match failed")
        mast_table = Observations.query_criteria(
            target_name=self.ticid, provenance_name="CDIPS"
        )
        if len(mast_table) == 0:
            raise ValueError("No CDIPS lightcurve in MAST.")
        else:
            print(f"Found {len(mast_table)} CDIPS lightcurves.")
        return mast_table.to_pandas()

    def get_cdips_url(self):
        """
        Each target is stored in a sub-directory based on the Sector it was observed in
        as a 4-digit zero-padded number.  They are further divided into sub-directories
        based on the camera and chip number they are on.  For example, 's0006/cam1_ccd1/' for
         Sector 6 light curves that are on CCD #1 on Camera #1.

        The light curves are in a `.fits` format familiar to users of the Kepler, K2,
        and TESS-short cadence light curves made by the NASA Ames team.  Their file names
        follow this convention:

        hlsp_cdips_tess_ffi_gaiatwo<gaiaid>-<sectornum>_tess_v01_llc.fits

        where:
          <gaiaid> = full Gaia DR2 target id, e.g., '0003321416308714545920'
          <sectornum? = 4-digit, zero-padded Sector number, e.g., '0006'
        """
        base = "https://archive.stsci.edu/hlsps/cdips/"
        assert self.sector is not None
        assert self.cam is not None
        assert self.ccd is not None
        assert self.gaiaid is not None
        sec = str(self.sector).zfill(4)
        gid = str(self.gaiaid).zfill(22)
        fp = (
            base
            + f"s{sec}/cam{self.cam}_ccd{self.ccd}"
            + f"/hlsp_cdips_tess_ffi_gaiatwo{gid}-"
            + f"{sec}-cam{self.cam}-ccd{self.ccd}"
            + "_tess_v01_llc.fits"
        )
        return fp

    def get_cdips_fits(self):
        """get cdips target and light curve header and data
        """
        fp = self.get_cdips_url()
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
            self.catalog_ref = self.header0["CDIPSREF"]
            self.catalog_gaiaids = self.header0["CDEXTCAT"]
            if self.verbose:
                print(self.header0[20:38])
                print(self.header0[-45:-25])
            return lc_data, lc_header

        except Exception:
            msg = f"File not found:\n{fp}\n"
            # msg += f"Using sector={self.sector} in {self.all_sectors}.\n"
            raise ValueError(msg)

    def validate_target_header(self):
        """
        see self.header0[20:38], [-45:-25] and self.header0['CDIPSREF']
        for useful target information
        """
        raise NotImplementedError()

    def get_cdips_lc(self, lc_type=None, aper_idx=None, sort=True):
        """
        Parameters
        ----------
        lc_type : str
            lightcurve type: [flux,tfa,pca,mag]
        aper_idx : int
            aperture [1,2,3] are [1,1.5,2.25] pix in radius
        normalize
        """
        aper = aper_idx if aper_idx is not None else self.aper_idx
        lctype = lc_type if lc_type is not None else self.lctype

        if lctype == "mag":
            # magnitude
            typstr1 = "IRM"
            typstr2 = "IRE"
        elif lctype == "tfa":
            # detrended light curve found by applying TFA with a fixed number of template stars
            typstr1 = "TFA"
            typstr2 = "IRE"
        elif lctype == "pca":
            # detrended light curve that regresses against the number of
            # principal components noted in the light curve's header
            typstr1 = "PCA"
            typstr2 = "IRE"
        else:
            # instrumental flux measured from differenced images
            typstr1 = "IFL"
            typstr2 = "IFE"
        time = self.data["TMID_BJD"]  # exposure mid-time at
        flux = self.data[f"{typstr1}{str(aper)}"]
        err = self.data[f"{typstr2}{str(aper)}"]
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
            time_format="jd",  # TIMEUNIT is bjd in fits header
            time_scale="tdb",  # TIMESYS in fits header
            centroid_col=None,
            centroid_row=None,
            quality=None,
            quality_bitmask=self.quality_bitmask,
            cadenceno=None,
            sector=self.sector,
            camera=self.cam,
            ccd=self.ccd,
            targetid=self.toi_params["TIC ID"]
            if self.toi_params is not None
            else self.ticid,
            ra=self.target_coord.ra.deg,
            dec=self.target_coord.dec.deg,
            label=None,
            meta=None,
        ).normalize()

    def get_aper_mask_cdips(self, sap_mask="round"):
        """
        This is an estimate of CDIPS aperture since
        self.hdulist[1].data.names does not contain aperture
        """
        aper_pix = CDIPS_APER_PIX[int(self.aper_idx) - 1]  # aper_idx=(1,2,3)
        print(
            f"CDIPS has no aperture info in fits. Estimating aperture instead using aper_idx={aper_pix} pix."
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
            self.tpf_tesscut,
            sap_mask=sap_mask,
            aper_radius=CDIPS_APER_PIX[idx],
        )
        self.aper_mask = aper_mask
        return aper_mask

    def plot_all_lcs(self):
        """
        """
        cdips_lcs = {}
        fig, ax = pl.subplots(1, 1, figsize=(10, 6))
        for aper in [1, 2, 3]:
            lc = self.get_cdips_lc(aper_idx=aper)
            lc.plot(ax=ax, label=f"aper={aper}")
            cdips_lcs[aper] = lc
        ax.set_title(f"{self.target_name} (sector {self.sector})")
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
                lc.time.value, period=period, t0=epoch, dur=duration / 24
            )
        else:
            tmask = np.zeros_like(lc.time.value, dtype=bool)
        # dummy holder
        flat, trend = lc.flatten(return_trend=True)
        # flatten using wotan
        wflat, wtrend = flatten(
            lc.time.value,
            lc.flux.value,
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
                lc.time.value, period=period, t0=epoch, dur=duration / 24
            )
        else:
            tmask = np.zeros_like(lc.time.value, dtype=bool)
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
                flat.time, period=period, t0=epoch, dur=duration / 24
            )
        else:
            tmask2 = np.zeros_like(lc.time.value, dtype=bool)
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
        fig = plot_odd_even(flat, period=period, epoch=epoch, yline=ylim)
        fig.suptitle(f"{self.target_name} (sector {flat.sector})")
        return fig

    def get_transit_mask(self, lc, period, epoch, duration_hours):
        """
        """
        tmask = get_transit_mask(
            lc.time.value, period=period, t0=epoch, dur=duration_hours / 24
        )
        return tmask


def get_cdips_inventory(fp=None, verbose=True, clobber=False):
    if fp is None:
        fp = join(DATA_PATH, "cdips_fits_inventory.txt")
    if not exists(fp) or clobber:
        url = "https://archive.stsci.edu/hlsps/cdips/cdips_inventory.txt"
        db = pd.read_csv(url, squeeze=True, names=["url"])
        db.to_csv(fp)
        msg = f"Saved: {fp}"
    else:
        db = pd.read_csv(fp, squeeze=True, names=["url"])
        msg = f"Loaded: {fp}"
    if verbose:
        print(msg)
    return db


def get_url_in_cdips_inventory(
    gaiaid, fp=None, verbose=True, clobber=False, sector=None
):
    if fp is None:
        fp = join(DATA_PATH, "cdips_fits_inventory.txt")
    db = get_cdips_inventory(fp=fp, verbose=verbose, clobber=clobber)
    # parse gaiaid in text
    gaiaids = db.apply(lambda x: x.split("_")[5].split("-")[0][7:])
    # check if gaia id matches any string
    idx = [True if str(gaiaid) in s else False for s in gaiaids]
    if verbose:
        print(f"There are {sum(idx)} CDIPS fits files found.")
    urls = db.loc[idx].values
    if len(urls) > 0:
        if sector is not None:
            if len(urls) > 0:
                n = 0
                for url in urls:
                    sec, cam, ccd = get_sector_cam_ccd_from_url(url)
                    if sec == int(sector):
                        return urls[n]
                    n += 1
            else:
                sec, cam, ccd = get_sector_cam_ccd_from_url(urls)
                if sec == int(sector):
                    return urls
        else:
            return urls
    else:
        return None


def get_sector_cam_ccd_from_url(url):
    sec = int(url.split("/")[1][1:])
    cam = int(url.split("/")[2][3])
    ccd = int(url.split("/")[2][8])
    return sec, cam, ccd

# -*- coding: utf-8 -*-

r"""
Multi-Sector Light Curves From TESS Full Frame Images (DIAMANTE)
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

# Import from package
from chronos.config import DATA_PATH
from chronos.target import Target
from chronos.tpf import FFI_cutout
from chronos.utils import (
    get_tois,
    get_transit_mask,
    parse_aperture_mask,
    TessLightCurve,
)

log = logging.getLogger(__name__)

__all__ = ["Diamante"]


class Diamante(Target):
    """
    https://archive.stsci.edu/hlsp/diamante
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
        quality_bitmask=None,
        search_radius=3,
        mission="tess",
        aper=1,
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
        """Initialize Diamante

        Attributes
        ----------
        lctype : str
            KSPSAP : Normalized light curve detrended by kepler spline
        aper : str
            best, small, large
        """
        self.base_url = "https://archive.stsci.edu/hlsps/diamante"
        self.diamante_catalog = self.get_diamante_catalog()
        self.new_diamante_candidates = self.get_new_diamante_candidates()
        self.sectors = self.all_sectors

        if self.gaiaid is None:
            _ = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)

        self.lc_num = lc_num
        self.lc_nums = [1, 2]
        if self.lc_num not in self.lc_nums:
            raise ValueError(f"Type not among {self.lc_nums}")
        self.aper = aper
        self.apers = [1, 2]
        if self.aper not in self.apers:
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
        # self.sap_mask = "round"
        # self.threshold_sigma = 5  # dummy
        # self.percentile = 95  # dummy
        # self.cutout_size = (15, 15)  # dummy
        # self.aper_radius = None
        # self.tpf_tesscut = None
        # self.ffi_cutout = None
        # self.aper_mask = None
        # self.contratio = None

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

    def get_diamante_lc(self, lc_num=None, aper=None, sort=True):
        """
        Parameters
        ----------
        lc_type : int

        """
        aper = self.aper if aper is None else aper
        lc_num = self.lc_num if lc_num is None else lc_num
        assert lc_num in self.lc_nums
        assert aper in self.apers

        if self.verbose:
            print(f"Using DIAmante LC{lc_num} (rad={aper}) lightcurve.")

        time = self.data["BTJD"] + 2457000  # BJD, days
        flux = self.data[f"LC{lc_num}_AP{aper}"]
        err = self.data[f"ELC{lc_num}_AP{aper}"]
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
            targetid=self.toi_params["TIC ID"]
            if self.toi_params is not None
            else self.ticid,
            ra=self.target_coord.ra.deg,
            dec=self.target_coord.dec.deg,
            label=None,
            meta=None,
        ).normalize()

    def validate_target_header(self):
        """
        see self.header0
        """
        raise NotImplementedError()

    def get_aper_mask_diamante(self, sap_mask="round"):
        """
        This is an estimate of QLP aperture based on
        self.hdulist[1].header['BESTAP']

        See:
        """
        rad = float(self.header["BESTAP"].split(":")[0])
        self.aper_radius = round(rad)
        print(f"Estimating QLP aperture using r={rad} pix.")
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

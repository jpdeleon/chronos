# -*- coding: utf-8 -*-

r"""
classes for working with lightcurves from the TESS-Gaia lightcurve pipeline:
https://archive.stsci.edu/hlsp/tglc
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

# Import from package
from chronos.config import DATA_PATH
from chronos.target import Target
from chronos.tpf import FFI_cutout
from chronos.utils import (
    get_sector_cam_ccd,
    parse_aperture_mask,
    TessLightCurve,
)
from chronos.constants import TESS_TIME_OFFSET

TGLC_SECTORS = np.arange(1, 3, 1)
TGLC_PAPER = "https://ui.adsabs.harvard.edu/abs/2023AJ....165...71H/abstract"
TGLC_README = "https://archive.stsci.edu/hlsps/tglc/hlsp_tglc_tess_ffi_all_tess_v1_readme.txt"

log = logging.getLogger(__name__)

__all__ = ["TGLC"]


class TGLC(Target):
    def __init__(
        self,
        sector=None,
        cam=None,
        ccd=None,
        name=None,
        toiid=None,
        ticid=None,
        epicid=None,
        gaiaDR3id=None,
        ra_deg=None,
        dec_deg=None,
        quality_bitmask=None,
        search_radius=3,
        lctype="psf",
        mission="tess",
        verbose=True,
        clobber=False,
    ):
        super().__init__(
            name=name,
            toiid=toiid,
            ticid=ticid,
            epicid=epicid,
            gaiaDR3id=gaiaDR3id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            search_radius=search_radius,
            verbose=verbose,
        )
        """Initialize TGLC.
        See http://archive.stsci.edu/hlsp/tglc
        Attributes
        ----------
        lctype: str
            TGLC lc types: ["psf", "aperture"]
        """
        if self.verbose:
            print("Using TGLC lightcurve.")
        self.sector = sector
        if self.sector is None:
            print(f"Available sectors: {self.all_sectors}")
            if len(self.all_sectors) != 1:
                idx = [
                    True if s in TGLC_SECTORS else False
                    for s in self.all_sectors
                ]
                if sum(idx) == 0:
                    msg = f"TGLC lc is currently available for sectors={TGLC_SECTORS}\n"
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
                        f"TGLC lc may be available for sectors {self.all_sectors[idx]}"
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

        self.lctype = lctype
        self.lctypes = ["psf", "aperture"]
        self.fits_url = None
        self.header0 = None  # target header
        self.hdulist = None
        self.data, self.header = self.get_tglc_fits()
        self.quality_bitmask = quality_bitmask
        self.lc = self.get_tglc_lc()
        self.tpf_tesscut = None
        self.ffi_cutout = None
        self.aper_mask = None

    def get_mast_table(self):
        """https://archive.stsci.edu/hlsp/cdips"""
        if self.gaia_params is None:
            _ = self.query_gaia_dr2_catalog(
                return_nearest_xmatch=True, version=3
            )
        if self.tic_params is None:
            _ = self.query_tic_catalog(return_nearest_xmatch=True)
        if not self.validate_gaia_tic_xmatch():
            raise ValueError("Gaia and Tic Catalog match failed")
        mast_table = Observations.query_criteria(
            target_name=self.ticid, provenance_name="TGLC"
        )
        if len(mast_table) == 0:
            raise ValueError("No TGLC lightcurve in MAST.")
        else:
            print(f"Found {len(mast_table)} TGLC lightcurves.")
        return mast_table.to_pandas()

    def get_tglc_url(self):
        """
        Each target has a FITS and TXT version of the light curves available.
        The files are stored in sub-directories based on the Sector they are
        in as a 4-digit, zero-padded number, e.g., "s0001/" for Sector 1.
        The data file naming convention is:

        hlsp_tglc_tess_ffi_gaiaid-<gaiaid>-s<sector>-cam<cam>-ccd<ccd>_tess_v1_llc.fits

        Ex:
        https://archive.stsci.edu/hlsps/tglc/s0001/cam1-ccd1/
        0064/7951/7581/2132/
        hlsp_tglc_tess_ffi_gaiaid-6479517581213243264-s0001-cam1-ccd1_tess_v1_llc.fits

        """
        base = "https://archive.stsci.edu/hlsps/tglc/"
        assert self.sector is not None
        assert self.gaiaid is not None
        assert self.cam is not None
        assert self.ccd is not None
        gaiaid = str(self.gaiaid).zfill(19)
        sect = str(self.sector).zfill(4)
        url = (
            base
            + f"s{sect}/"
            + f"cam{self.cam}-ccd{self.ccd}/"
            + f"00{gaiaid[0:2]}/{gaiaid[2:6]}/{gaiaid[6:10]}/{gaiaid[10:14]}/"
            + "hlsp_tglc_tess_ffi_gaiaid-"
            + f"{gaiaid}-s{sect}-cam{self.cam}-"
            + f"ccd{self.ccd}_tess_v1_llc.fits"
        )
        return url

    def get_tglc_fits(self):
        """get tglc target and light curve header and data"""
        fp = self.get_tglc_url()
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

    def get_tglc_lc(self, lctype=None, aper_idx=None, sort=True):
        """
        Parameters
        ----------
        """
        # aper = aper_idx if aper_idx is not None else self.aper_idx
        lctype = lctype if lctype is not None else self.lctype

        tstr = "time"
        fstr = f"cal_{lctype}_flux"
        if lctype == "psf":
            estr = "CPSF_ERR"  # raw flux err in header
        elif lctype == "corr":
            estr = "CAPE_ERR"
        else:
            raise ValueError(" or ".join(self.lctypes))
        # barycentric-corrected, truncated TESS Julian Date (BJD - 2457000.0)
        time = self.data[tstr]
        flux = self.data[fstr]
        err = np.zeros_like(time) + self.header[estr]
        # background = self.data["background"]
        tess_flag = self.data["TESS_flags"]
        tglc_flag = self.data["TGLC_flags"]
        flag = list(tess_flag == 0) and list(tglc_flag == 0)
        if sort:
            idx = np.argsort(time)
        else:
            idx = np.ones_like(time, bool)
        # hack tess lightkurve
        return TessLightCurve(
            time=time[idx][flag] + TESS_TIME_OFFSET,
            flux=flux[idx][flag],
            flux_err=err[idx][flag],
            # FIXME: only day works when using lc.to_periodogram()
            time_format="jd",  # TIMEUNIT is bjd in fits header
            time_scale="tdb",  # TIMESYS in fits header
            # centroid_col=ypos,
            # centroid_row=xpos,
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
        )  # .normalize()

    def get_aper_mask_tglc(self, sap_mask="square"):
        """
        This is an estimate of TGLC aperture only
        """
        print(
            f"TGLC's {self.lctype} light curve has no aperture info in fits. Representing the aperture using rad=3 pix instead."
        )
        if self.ffi_cutout is None:
            # first download tpf cutout
            self.ffi_cutout = FFI_cutout(
                sector=self.sector,
                gaiaDR3id=self.gaiaid,
                toiid=self.toiid,
                ticid=self.ticid,
                search_radius=self.search_radius,
                quality_bitmask=self.quality_bitmask,
            )
        self.tpf_tesscut = self.ffi_cutout.get_tpf_tesscut()
        aper_mask = parse_aperture_mask(
            self.tpf_tesscut, sap_mask=sap_mask, aper_radius=3
        )
        self.aper_mask = aper_mask
        return aper_mask

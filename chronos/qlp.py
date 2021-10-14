# -*- coding: utf-8 -*-

r"""
classes for working with lightcurves from the QLP pipeline:
http://archive.stsci.edu/hlsp/qlp
"""

# Import standard library
from pathlib import Path
import logging

# Import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import astropy.units as u
import lightkurve as lk
from astropy.io import fits

# Import from package
from chronos.config import DATA_PATH
from chronos.target import Target
from chronos.tpf import FFI_cutout

# from chronos.plot import plot_tls, plot_odd_even
from chronos.utils import get_transit_mask, parse_aperture_mask, TessLightCurve


log = logging.getLogger(__name__)

__all__ = ["QLP"]

QLP_SECTORS = np.arange(11, 27, 1)


class QLP(Target):
    """
    http://archive.stsci.edu/hlsp/qlp
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
        aper="best",
        lctype="KSPSAP",
        mission="tess",
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
        """Initialize QLP.
        See http://archive.stsci.edu/hlsp/qlp

        Attributes
        ----------
        lctype : str
            KSPSAP : Normalized light curve detrended by kepler spline
        aper : str
            best, small, large
        """
        self.sector = sector
        if self.sector is None:
            print(f"Available sectors: {self.all_sectors}")
            if len(self.all_sectors) != 1:
                idx = [
                    True if s in QLP_SECTORS else False
                    for s in self.all_sectors
                ]
                if sum(idx) == 0:
                    msg = f"QLP lc is currently available for sectors={QLP_SECTORS}\n"
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
                        f"QLP lc may be available for sectors {self.all_sectors[idx]}"
                    )
            print(f"Using sector={self.sector}.")

        if self.gaiaid is None:
            _ = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)

        self.aper = aper
        self.apers = ["best", "small", "large"]
        if self.aper not in self.apers:
            raise ValueError(f"Type not among {self.apers}")
        self.quality_bitmask = quality_bitmask
        self.fits_url = None
        self.hdulist = None
        self.header0 = None
        self.lctype = lctype.upper()
        self.lctypes = ["SAP", "KSPSAP"]
        if self.lctype not in self.lctypes:
            raise ValueError(f"Type not among {self.lctypes}")
        self.data, self.header = self.get_qlp_fits()
        self.lc = self.get_qlp_lc()
        self.lc.targetid = self.ticid
        self.cadence = self.header["TIMEDEL"] * u.d
        self.time = self.lc.time.value
        self.flux = self.lc.flux.value
        self.err = self.lc.flux_err.value
        self.sap_mask = "round"
        self.threshold_sigma = 5  # dummy
        self.percentile = 95  # dummy
        self.cutout_size = (15, 15)  # dummy
        self.aper_radius = None
        self.tpf_tesscut = None
        self.ffi_cutout = None
        self.aper_mask = None
        self.contratio = None

    def get_qlp_url(self):
        """
        hlsp_qlp_tess_ffi_<sector>-<tid>_tess_v01_llc.<exten>
        where:

        <sector> = The Sector represented as a 4-digit, zero-padded string,
                    preceded by an 's', e.g., 's0026' for Sector 26.
        <tid> = The full, 16-digit, zeo-padded TIC ID.
        <exten> = The light curve data type, either "fits" or "txt".
        """
        base = "https://archive.stsci.edu/hlsps/qlp/"
        assert self.sector is not None
        sec = str(self.sector).zfill(4)
        tic = str(self.ticid).zfill(16)
        fp = (
            base
            + f"s{sec}/{tic[:4]}/{tic[4:8]}/{tic[8:12]}/{tic[12:16]}/hlsp_qlp_tess_ffi_s{sec}-{tic}_tess_v01_llc.fits"
        )
        return fp

    def get_qlp_fits(self):
        """get qlp target and light curve header and data
        """
        fp = self.get_qlp_url()
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

    def get_qlp_lc(self, lc_type=None, aper=None, sort=True):
        """
        Parameters
        ----------
        lc_type : str
            {SAP, KSPSAP}
        """
        lc_type = lc_type.upper() if lc_type is not None else self.lctype
        aper = aper.upper() if aper is not None else self.aper
        assert lc_type in self.lctypes
        assert aper in self.apers

        if self.verbose:
            print(f"Using QLP {lc_type} (rad={self.aper}) lightcurve.")

        time = self.data["TIME"] + 2457000  # BJD, days
        if aper == "small":
            flux = self.data["KSPSAP_FLUX_SML"]
        elif aper == "large":
            flux = self.data["KSPSAP_FLUX_LAG"]
        else:
            flux = self.data[f"{lc_type}_FLUX"]
        if lc_type == "KSPSAP":
            err = self.data[f"{lc_type}_FLUX_ERR"]
        else:
            err = np.ones_like(flux) * np.std(flux)
        x = self.data["SAP_X"]
        y = self.data["SAP_Y"]
        quality = self.data["QUALITY"]
        cadence = self.data["CADENCENO"]
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
            centroid_col=x,
            centroid_row=y,
            quality=quality,
            quality_bitmask=self.quality_bitmask,
            cadenceno=cadence,
            sector=self.sector,
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

    def get_aper_mask_qlp(self, sap_mask="round"):
        """
        This is an estimate of QLP aperture based on
        self.hdulist[1].header['BESTAP']

        See:
        https://archive.stsci.edu/hlsps/qlp/hlsp_qlp_tess_ffi_all_tess_v1_data-prod-desc.pdf
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

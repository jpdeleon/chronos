# -*- coding: utf-8 -*-

r"""
classes for searching target and querying cluster catalogs

See also from astroquery.xmatch import XMatch
"""
# Import standard library
from os.path import join, exists
import logging

# Import library
import numpy as np
import pandas as pd
import astropy.units as u
from lightkurve import TessLightCurve
from astropy.io import fits

# Import from package
from chronos.config import DATA_PATH
from chronos.target import Target

log = logging.getLogger(__name__)

__all__ = ["CDIPS", "get_cdips_inventory", "get_url_in_cdips_inventory"]

CDIPS_SECTORS = [6, 7, 8, 9, 10, 11]
CDIPS_APER_PIX = [1, 1.5, 2.25]


class _TessLightCurve(TessLightCurve):
    """augments parent class by adding convenience methods"""

    def detrend(self, break_tolerance=None):
        lc = self.copy()
        half = lc.time.shape[0] // 2
        if half % 2 == 0:
            # add 1 if even
            half += 1
        return lc.flatten(
            window_length=half, polyorder=1, break_tolerance=break_tolerance
        )


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
        search_radius=2 * u.arcsec,
        verbose=True,
        lctype="flux",
        aper_idx=1
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
        self.sector = sector
        self.all_sectors = self.get_all_sectors()

        if self.sector is None:
            self.sector = self.all_sectors[0]
            print(f"Available sectors: {self.all_sectors}")
            print(f"Using sector={self.sector}.")
        msg = f"CDIPS lc is currently available for sectors={CDIPS_SECTORS}\n"
        assert self.sector in CDIPS_SECTORS, msg

        self.cam = cam
        self.ccd = ccd
        if self.gaiaid is None:
            _ = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)
        if not np.all([self.sector, self.cam, self.ccd]):
            # overwrite
            sector, cam, ccd = self.get_sector_cam_ccd()
            self.sector = sector if self.sector is None else str(self.sector)
            self.cam = cam if self.cam is None else str(self.cam)
            self.ccd = ccd if self.ccd is None else str(self.ccd)
        # self.mission = mission
        self.lctype = lctype
        self.lctypes = ["flux", "mag", "tfa", "pca"]
        self.aper_idx = str(aper_idx)
        assert self.aper_idx in [
            "1",
            "2",
            "3",
        ], "CDIPS has only [1,2,3] aperture indices"
        self.header0 = None  # target header
        self.catalog_ref = None  # references
        self.catalog_gaiaids = None  # gaia id(s) in catalog_ref
        self.hdulist = None
        # self.ccd_info = Tesscut.get_sectors(self.target_coord).to_pandas()
        self.data, self.header = self.get_cdips_fits()
        time, flux, err = self.get_cdips_lc()
        self.quality_bitmask = quality_bitmask
        # hack tess lightkurve
        self.lc = _TessLightCurve(
            time=time,
            flux=flux,
            flux_err=err,
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
        self.lc.targetid = self.ticid
        self.cadence = self.header["XPOSURE"] * u.second  # .to(u.minute)
        self.time = self.lc.time
        self.flux = self.lc.flux
        self.err = self.lc.flux_err
        if self.lctype not in self.lctypes:
            raise ValueError(f"Type not among {self.lctypes}")
        self.fits_url = None
        self.readme_url = "https://archive.stsci.edu/hlsps/cdips/hlsp_cdips_tess_ffi_all_tess_v01_readme.md"

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
        sec = self.sector.zfill(4)
        fp = (
            base
            + f"s{sec}/cam{self.cam}_ccd{self.ccd}"
            + f"/hlsp_cdips_tess_ffi_gaiatwo000{self.gaiaid}-"
            + f"{sec}-cam{self.cam}-ccd{self.ccd}"
            + f"_tess_v01_llc.fits"
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
            msg += f"Using sector={self.sector} in {self.all_sectors}.\n"
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
        t = self.data["TMID_BJD"]  # exposure mid-time at
        f = self.data[f"{typstr1}{str(aper)}"]
        e = self.data[f"{typstr2}{str(aper)}"]
        if sort:
            idx = np.argsort(t)
        else:
            idx = np.ones_like(t, bool)
        # use lightkurve's normalize method instead
        # if normalize:
        #     if lctype != "flux":
        #         raise ValueError("Cannot normalize magnitude")
        #     else:
        #         f /= np.median(f)
        #         e = e/f
        return (t[idx], f[idx], e[idx])


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

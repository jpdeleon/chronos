# -*- coding: utf-8 -*-

r"""
classes for searching k2 lightcurves produced by EVEREST and K2SFF pipelines
"""
# Import standard library
from os.path import join, exists
import logging

# Import library
import numpy as np
import pandas as pd
import astropy.units as u
from lightkurve import KeplerLightCurve, KeplerQualityFlags
from astropy.io import fits

# Import from package
from chronos.config import DATA_PATH
from chronos.search import Target

log = logging.getLogger(__name__)

__all__ = ["Everest"]

K2_TIME_OFFSET = 2454833


class Everest(Target):
    """
    """

    def __init__(
        self,
        campaign=None,
        name=None,
        epicid=None,
        gaiaDR2id=None,
        ra_deg=None,
        dec_deg=None,
        quality_bitmask="default",
        verbose=True,
        flux_type="flux",  # or fcor
    ):
        super().__init__(
            name=name,
            epicid=epicid,
            gaiaDR2id=gaiaDR2id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            verbose=verbose,
        )
        """Initialize Everest

        Attributes
        ----------

        """
        self.campaign = campaign
        self.flux_type = flux_type.upper()
        self.quality_bitmask = quality_bitmask
        url, filename = self.get_everest_url_and_fn()
        self.url = url
        self.filename = filename
        self.quality = None
        self.cadenceno = None
        time, flux, err = self.get_everest_lc()
        # hack
        self.lc = KeplerLightCurve(
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
        return_err=True,
    ):
        """
        see also https://archive.stsci.edu/k2/hlsp/everest/search.php

        flux_type : str
            'flux' = PLD-de-trended flux; 'fcor' = de-trended flux with CBV correction
        normalize : bool
            divide flux (and err) by its median
        quality: str
            option to choose which cadences will be masked
        return_err : bool
            returns time, flux, err if True; time and flux only otherwise
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
            idx = np.isfinite(time) & np.isfinite(flux)
            time, flux, err = time[idx], flux[idx], err[idx]
            if normalize:
                err /= np.median(flux)  # divide by median of raw flux
                flux /= np.median(flux)
            time += K2_TIME_OFFSET
            if return_err:
                return (time, flux, err)
            else:
                return (time, flux)
        except Exception as e:
            print(e)

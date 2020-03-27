# -*- coding: utf-8 -*-

r"""
classes for k2 lightcurves produced by EVEREST and K2SFF pipelines
"""
# Import standard library
from os.path import join, exists
import logging

# Import library
import numpy as np
import pandas as pd
import astropy.units as u
import lightkurve as lk
from astropy.io import fits

# Import from package
from chronos.config import DATA_PATH
from chronos.target import Target
from chronos.constants import K2_TIME_OFFSET
from chronos.utils import detrend, get_all_campaigns

log = logging.getLogger(__name__)

__all__ = ["K2", "Everest", "K2sff"]


class _KeplerLightCurve(lk.KeplerLightCurve):
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
        sap_mask=None,
        aper_radius=None,
        percentile=None,
        threshold_sigma=None,
        use_pld=True,
    ):
        NotImplementedError
        # lc = tpf.to_lightcurve()


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
        time, flux, err = self.get_everest_lc()
        # hack
        self.lc_everest = _KeplerLightCurve(
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
        time, flux = self.get_k2sff_lc()
        # hack
        self.lc_k2sff = _KeplerLightCurve(
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

    def get_k2sff_lc(self, campaign=None, flux_type="fcor", normalize=True):
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
            with fits.open(url) as hl:
                if self.verbose:
                    print(hl.info())
                recarray = hl[1].data
                cols = recarray.columns.names
                assert (
                    flux_type in cols
                ), f"flux_type={flux_type} not in {cols}"
                time = recarray["T"]
                flux = recarray[flux_type]
            idx = np.isfinite(time) & np.isfinite(flux)
            time, flux = time[idx], flux[idx]
            if normalize:
                flux /= np.median(flux)
            time += K2_TIME_OFFSET
            return time, flux
        except Exception as e:
            print(e)

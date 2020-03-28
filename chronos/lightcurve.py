# -*- coding: utf-8 -*-

r"""
classes for manipulating lightcurve
"""
# Import standard library
from os.path import join, exists
import logging

# Import library
# from matplotlib.figure import Figure
# from matplotlib.image import AxesImage
# from loguru import logger
import getpass
import numpy as np
import astropy.units as u

# from scipy.signal import detrend
from astropy.timeseries import LombScargle
from astropy.io import fits
import lightkurve as lk

# Import from package
from chronos.config import DATA_PATH
from chronos.tpf import Tpf, FFI_cutout
from chronos.cdips import CDIPS
from chronos.utils import (
    remove_bad_data,
    parse_aperture_mask,
    get_fluxes_within_mask,
    detrend,
)

user = getpass.getuser()
MISSION = "TESS"
fitsoutdir = join("/home", user, "data/transit")

log = logging.getLogger(__name__)

__all__ = ["ShortCadence", "LongCadence"]


class LongCadence(FFI_cutout):
    """
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
        search_radius=3,
        sap_mask="square",
        aper_radius=1,
        threshold_sigma=5,
        percentile=95,
        cutout_size=(15, 15),
        quality_bitmask="default",
        apply_data_quality_mask=False,
        mission="tess",
        clobber=True,
        verbose=True,
        # mission="TESS",
        # quarter=None,
        # month=None,
        # campaign=None,
        # limit=None,
    ):
        """
        handles lightcurve creation and manipulation for TESS long cadence data
        using `FFI_cutout`

        Attributes
        ----------
        sap_mask : str
            aperture mask shape (default=square)
        aper_radius : int
            aperture radius
        threshold_sigma : float
            threshold sigma above median flux
        percentile : float
            percentile of flux
        quality_bitmask : str
            (default=default)
            https://github.com/KeplerGO/lightkurve/blob/master/lightkurve/utils.py#L210
        apply_data_quality_mask : bool (default=False)
            remove bad data identified in TESS Data Release notes

        """
        super().__init__(
            name=name,
            toiid=toiid,
            ticid=ticid,
            epicid=epicid,
            gaiaDR2id=gaiaDR2id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            sector=sector,
            search_radius=search_radius,
            sap_mask=sap_mask,
            aper_radius=aper_radius,
            threshold_sigma=threshold_sigma,
            percentile=percentile,
            cutout_size=cutout_size,
            quality_bitmask=quality_bitmask,
            apply_data_quality_mask=apply_data_quality_mask,
            verbose=verbose,
            clobber=clobber,
        )
        self.corrector = None
        self.lc_custom = None
        self.lc_custom_raw = None
        self.lc_cdips = None
        self.contratio = None
        self.cdips = None

    def make_custom_lc(
        self,
        sector=None,
        tpf_size=None,
        sap_mask=None,
        aper_radius=None,
        percentile=None,
        threshold_sigma=None,
        use_pld=True,
        pixel_components=3,
        spline_n_knots=100,
        spline_degree=3,
        background_mask=None,
        pca_nterms=5,
        with_offset=True,
    ):
        """
        create a custom lightcurve based on this tutorial:
        https://docs.lightkurve.org/tutorials/04-how-to-remove-tess-scattered-light-using-regressioncorrector.html

        Parameters
        ----------
        sector : int or str
            specific sector or all
        cutout_size : tuple
            tpf cutout size
        aper_radius: int
            aperture mask radius
        percentile: float
            aperture mask percentile
        threshold_sigma: float
            aperture mask threshold [sigma]
        method : float
            PLD (default)

        Returns
        -------
        corrected_lc : lightkurve object
        """
        sector = sector if sector is not None else self.sector
        sap_mask = sap_mask if sap_mask else self.sap_mask
        aper_radius = aper_radius if aper_radius else self.aper_radius
        percentile = percentile if percentile else self.percentile
        threshold_sigma = (
            threshold_sigma if threshold_sigma else self.threshold_sigma
        )
        cutout_size = tpf_size if tpf_size else self.cutout_size

        tpf_tesscut = self.get_tpf_tesscut(
            sector=sector, cutout_size=cutout_size
        )

        self.aper_mask = parse_aperture_mask(
            tpf_tesscut,
            sap_mask=sap_mask,
            aper_radius=aper_radius,
            percentile=percentile,
            threshold_sigma=threshold_sigma,
            verbose=False,
        )

        raw_lc = tpf_tesscut.to_lightcurve(
            method="aperture", aperture_mask=self.aper_mask
        )
        # remove nans
        idx = (
            np.isnan(raw_lc.time)
            | np.isnan(raw_lc.flux)
            | np.isnan(raw_lc.flux_err)
        )
        self.tpf_tesscut = tpf_tesscut[~idx]
        self.lc_custom_raw = raw_lc[~idx]

        if use_pld:
            if self.verbose:
                print("Removing scattered light + applying PLD")
            pld = lk.TessPLDCorrector(
                self.tpf_tesscut, aperture_mask=self.aper_mask
            )
            if background_mask is None:
                background_mask = ~self.aper_mask
            corrected_lc = pld.correct(
                pixel_components=pixel_components,
                spline_n_knots=spline_n_knots,
                spline_degree=spline_degree,
                background_mask=background_mask,
            )
            self.corrector = pld
        else:
            if self.verbose:
                print("Removing scattered light")
            # Make a design matrix and pass it to a linear regression corrector
            regressors = tpf_tesscut.flux[~idx][:, ~self.aper_mask]
            dm = (
                lk.DesignMatrix(regressors, name="regressors")
                .pca(nterms=pca_nterms)
                .append_constant()
            )
            rc = lk.RegressionCorrector(raw_lc)
            self.corrector = rc
            corrected_lc = rc.correct(dm)

            # Optional: Remove the scattered light, allowing for the large offset from scattered light
            if with_offset:
                corrected_lc = (
                    raw_lc - rc.model_lc + np.percentile(rc.model_lc.flux, q=5)
                )
        lc = corrected_lc.normalize()
        self.lc_custom = lc

        # compute Contamination
        if self.gaia_sources is None:
            gaia_sources = self.query_gaia_dr2_catalog(radius=120)
        else:
            gaia_sources = self.gaia_sources
        fluxes = get_fluxes_within_mask(
            self.tpf_tesscut, self.aper_mask, gaia_sources
        )
        self.contratio = sum(fluxes) - 1
        return lc

    def get_cdips_lc(
        self, sector=None, aper_idx=3, lctype="flux", verbose=False
    ):
        verbose = verbose if verbose is not None else self.verbose
        sector = sector if sector is not None else self.sector
        if self.gaiaid is None:
            d = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)
            self.gaiaid = int(d.source_id)
        cdips = CDIPS(
            gaiaDR2id=self.gaiaid,
            sector=sector,
            aper_idx=aper_idx,
            lctype=lctype,
            verbose=verbose,
        )
        self.cdips = cdips
        self.lc_cdips = cdips.lc
        self.lc_cdips.targetid = self.ticid
        return cdips.lc


class ShortCadence(Tpf):
    """
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
        search_radius=3,
        sap_mask="pipeline",
        aper_radius=1,
        threshold_sigma=5,
        percentile=95,
        quality_bitmask="default",
        apply_data_quality_mask=False,
        apphot_method="aperture",  # or prf
        clobber=True,
        verbose=True,
        # mission="TESS",
        # quarter=None,
        # month=None,
        # campaign=None,
        # limit=None,
    ):
        """
        sap_mask : str
            aperture mask shape (default=pipeline)
        aper_radius : int
            if aperture radius for mask!=pipeline
        threshold_sigma : float
            threshold sigma above median flux for mask!=pipeline
        percentile : float
            percentile of flux for mask!=pipeline
        quality_bitmask : str
            (default=default)
            https://github.com/KeplerGO/lightkurve/blob/master/lightkurve/utils.py#L210
        apply_data_quality_mask : bool (default=False)
            remove bad data identified in TESS Data Release notes
        """
        super().__init__(
            name=name,
            toiid=toiid,
            ticid=ticid,
            epicid=epicid,
            gaiaDR2id=gaiaDR2id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            sector=sector,
            search_radius=search_radius,
            sap_mask=sap_mask,
            aper_radius=aper_radius,
            threshold_sigma=threshold_sigma,
            percentile=percentile,
            quality_bitmask=quality_bitmask,
            apply_data_quality_mask=apply_data_quality_mask,
            verbose=verbose,
            clobber=clobber,
        )
        self.apphot_method = apphot_method
        self.lc_custom = None
        self.lc_custom_raw = None
        self.lcf = None
        self.lc_sap = None
        self.lc_pdcsap = None
        self.contratio = None

    def get_lc(self, lctype="pdcsap", sector=None, quality_bitmask=None):
        """
        """
        sector = sector if sector is not None else self.sector
        quality_bitmask = (
            quality_bitmask if quality_bitmask else self.quality_bitmask
        )
        if self.lcf is not None:
            # reload lcf if already in memory
            if self.lcf.sector == sector:
                lcf = self.lcf
            else:
                query_str = (
                    f"TIC {self.ticid}" if self.ticid else self.target_coord
                )
                if self.verbose:
                    print(
                        f"Searching lightcurvefile for {query_str} (sector {sector})"
                    )
                q = lk.search_lightcurvefile(
                    query_str, sector=sector, mission=MISSION
                )
                if len(q) == 0:
                    if self.verbose:
                        print(
                            f"Searching lightcurvefile for {self.target_coord.to_string()} (sector {sector})"
                        )
                    q = lk.search_lightcurvefile(
                        self.target_coord, sector=sector, mission=MISSION
                    )
                assert q is not None, "Empty result. Check long cadence."
                if self.verbose:
                    print(f"Found {len(q)} lightcurves")
                if (sector == "all") & (len(self.all_sectors) > 1):
                    lcf = q.download_all(quality_bitmask=quality_bitmask)
                else:
                    lcf = q.download(quality_bitmask=quality_bitmask)
                self.lcf = lcf
        else:
            query_str = (
                f"TIC {self.ticid}" if self.ticid else self.target_coord
            )
            if self.verbose:
                print(
                    f"Searching lightcurvefile for {query_str} (sector {sector})"
                )
            q = lk.search_lightcurvefile(
                query_str, sector=sector, mission=MISSION
            )
            if len(q) == 0:
                if self.verbose:
                    print(
                        f"Searching lightcurvefile for {self.target_coord.to_string()} (sector {sector})"
                    )
                q = lk.search_lightcurvefile(
                    self.target_coord, sector=sector, mission=MISSION
                )
            assert q is not None, "Empty result. Check long cadence."
            if self.verbose:
                print(f"Found {len(q)} lightcurves")
            if (sector == "all") & (len(self.all_sectors) > 1):
                lcf = q.download_all(quality_bitmask=quality_bitmask)
            else:
                lcf = q.download(quality_bitmask=quality_bitmask)
            self.lcf = lcf
        assert lcf is not None, "Empty result. Check long cadence."
        sap = lcf.SAP_FLUX
        pdcsap = lcf.PDCSAP_FLUX
        if isinstance(lcf, lk.LightCurveFileCollection):
            # merge multi-sector into one lc
            if len(lcf) > 1:
                sap0 = sap[0].normalize()
                sap = [sap0.append(l.normalize()) for l in sap[1:]][0]
                pdcsap0 = pdcsap[0].normalize()
                pdcsap = [pdcsap0.append(l.normalize()) for l in pdcsap[1:]][0]
            else:
                raise ValueError(
                    f"Only sector {lcf[0].sector} (in {self.all_sectors}) is available"
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
        sector=None,
        sap_mask=None,
        aper_radius=None,
        percentile=None,
        threshold_sigma=None,
        use_pld=True,
        pixel_components=3,
        spline_n_knots=100,
        spline_degree=3,
        background_mask=None,
        pca_nterms=5,
        with_offset=True,
    ):
        """
        create a custom lightcurve with background subtraction, based on this tutorial:
        https://docs.lightkurve.org/tutorials/04-how-to-remove-tess-scattered-light-using-regressioncorrector.html

        Parameters
        ----------
        sector : int or str
            specific sector or all
        aper_radius: int
            aperture mask radius
        percentile: float
            aperture mask percentile
        threshold_sigma: float
            aperture mask threshold [sigma]
        pca_nterms : int
            number of pca terms to use

        Returns
        -------
        corrected_lc : lightkurve object
        """
        sector = sector if sector is not None else self.sector
        sap_mask = sap_mask if sap_mask else self.sap_mask
        aper_radius = aper_radius if aper_radius else self.aper_radius
        percentile = percentile if percentile else self.percentile
        threshold_sigma = (
            threshold_sigma if threshold_sigma else self.threshold_sigma
        )
        if self.tpf is None:
            tpf, tpf_info = self.get_tpf(sector=sector, return_df=True)
        else:
            if self.tpf.sector == sector:
                tpf = self.tpf
            else:
                tpf, tpf_info = self.get_tpf(sector=sector, return_df=True)
        # Make an aperture mask and a raw light curve
        self.aper_mask = parse_aperture_mask(
            tpf,
            sap_mask=sap_mask,
            aper_radius=aper_radius,
            percentile=percentile,
            threshold_sigma=threshold_sigma,
            verbose=False,
        )
        raw_lc = tpf.to_lightcurve(
            method="aperture", aperture_mask=self.aper_mask
        )
        # remove nans
        idx = (
            np.isnan(raw_lc.time)
            | np.isnan(raw_lc.flux)
            | np.isnan(raw_lc.flux_err)
        )
        self.tpf = tpf[~idx]
        self.raw_lc = raw_lc[~idx]
        # add method
        raw_lc.detrend = lambda: detrend(raw_lc)

        if use_pld:
            if self.verbose:
                print("Removing scattered light + applying PLD")
            pld = lk.TessPLDCorrector(self.tpf, aperture_mask=self.aper_mask)
            if background_mask is None:
                background_mask = ~self.aper_mask
            corrected_lc = pld.correct(
                pixel_components=pixel_components,
                spline_n_knots=spline_n_knots,
                spline_degree=spline_degree,
                background_mask=background_mask,
            )
            self.corrector = pld
        else:
            if self.verbose:
                print("Removing scattered light")
            # Make a design matrix and pass it to a linear regression corrector
            regressors = tpf.flux[~idx][:, ~self.aper_mask]
            dm = (
                lk.DesignMatrix(regressors, name="pixels")
                .pca(pca_nterms)
                .append_constant()
            )

            # Regression Corrector Object
            rc = lk.RegressionCorrector(self.raw_lc)
            self.corrector = rc
            corrected_lc = rc.correct(dm)

            # Optional: Remove the scattered light, allowing for the large offset from scattered light
            if with_offset:
                corrected_lc = (
                    self.raw_lc
                    - rc.model_lc
                    + np.percentile(rc.model_lc.flux, q=5)
                )
        lc = corrected_lc.normalize()
        self.lc_custom = lc
        # compute Contamination
        if self.gaia_sources is None:
            gaia_sources = self.query_gaia_dr2_catalog(
                radius=120, verbose=False
            )
        else:
            gaia_sources = self.gaia_sources
        fluxes = get_fluxes_within_mask(self.tpf, self.aper_mask, gaia_sources)
        self.contratio = sum(fluxes) - 1
        tic_contratio = self.tic_params.contratio
        dcontratio = abs(tic_contratio-self.contratio)
        if (tic_contratio is not None) & (dcontratio>0.5):
            print(f"contratio: {self.contratio:.2f} (TIC={tic_contratio:.2f})")
        return lc


# class LightCurve(ShortCadence, LongCadence):
#     raise NotImplementedError

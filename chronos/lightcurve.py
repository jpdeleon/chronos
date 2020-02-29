# -*- coding: utf-8 -*-

r"""
classes for creating lightcurve
"""
# Import standard library
from os.path import join, exists
import logging

# Import library
import numpy as np
import astropy.units as u
from lightkurve import TessLightCurve
from astropy.io import fits
import lightkurve as lk

# Import from package
from chronos.config import DATA_PATH
from chronos.target import Target
from chronos.cdips import CDIPS
from chronos.utils import remove_bad_data, parse_aperture_mask
import getpass

user = getpass.getuser()
MISSION = "TESS"
fitsoutdir = join("/home", user, "data/transit")

log = logging.getLogger(__name__)

__all__ = ["ShortCadence", "LongCadence", "LightCurve"]


class LongCadence(Target):
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
        search_radius=3 * u.arcsec,
        sap_mask="threshold",
        aper_radius=1,
        threshold_sigma=3,
        percentile=90,
        cutout_size=(50, 50),
        quality_bitmask="default",
        apply_data_quality_mask=True,
        clobber=True,
        verbose=True,
        # mission="TESS",
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
            clobber=clobber,
        )

        # self.mission = mission
        self.sector = sector
        if self.sector is None:
            self.sector = self.get_all_sectors()[0]
            print(f"Available sectors: {self.get_all_sectors()}")
            print(f"Using sector={self.sector}.")
        self.sap_mask = sap_mask
        self.aper_mask = None
        self.aper_radius = aper_radius
        self.percentile = percentile
        self.threshold_sigma = threshold_sigma
        self.cutout_size = cutout_size
        self.search_radius = search_radius
        self.quality_bitmask = quality_bitmask
        self.apply_data_quality_mask = apply_data_quality_mask
        self.tpf_tesscut = None
        self.corrector = None
        self.lc_custom = None
        self.lc_custom_raw = None
        self.lc_cdips = None

    def get_tpf_tesscut(self, sector=None, cutout_size=None, verbose=True):
        """
        """
        cutout_size = cutout_size if cutout_size else self.cutout_size
        sector = sector if sector else self.sector
        if self.tpf_tesscut is None:
            if verbose:
                print("Searching targetpixelfile using Tesscut")
            tpf = lk.search_tesscut(self.target_coord, sector=sector).download(
                quality_bitmask=self.quality_bitmask, cutout_size=cutout_size
            )
        else:
            if (self.tpf_tesscut.sector == sector) & (
                self.cutout_size == cutout_size
            ):
                tpf = self.tpf_tesscut
            else:
                if verbose:
                    print("Searching targetpixelfile using Tesscut")
                tpf = lk.search_tesscut(
                    self.target_coord, sector=sector
                ).download(
                    quality_bitmask=self.quality_bitmask,
                    cutout_size=cutout_size,
                )
        assert tpf is not None, "No results from Tesscut search."
        # remove zeros
        zero_mask = (tpf.flux_err == 0).all(axis=(1, 2))
        if zero_mask.sum() > 0:
            tpf = tpf[~zero_mask]
        if self.apply_data_quality_mask:
            tpf = remove_bad_data(tpf, sector=sector, verbose=verbose)
        self.tpf_tesscut = tpf
        return tpf

    def get_aper_mask(
        self,
        sector=None,
        sap_mask=None,
        aper_radius=None,
        percentile=None,
        threshold_sigma=None,
        tpf_size=None,
        verbose=True,
    ):
        """
        """
        sector = sector if sector else self.sector
        sap_mask = sap_mask if sap_mask else self.sap_mask
        aper_radius = aper_radius if aper_radius else self.aper_radius
        percentile = percentile if percentile else self.percentile
        threshold_sigma = (
            threshold_sigma if threshold_sigma else self.threshold_sigma
        )
        cutout_size = tpf_size if tpf_size else self.cutout_size

        tpf = self.get_tpf_tesscut(
            sector=sector, cutout_size=cutout_size, verbose=verbose
        )

        aper_mask = parse_aperture_mask(
            tpf,
            sap_mask=sap_mask,
            aper_radius=aper_radius,
            percentile=percentile,
            threshold_sigma=threshold_sigma,
            verbose=verbose,
        )
        self.aper_mask = aper_mask
        return aper_mask

    def make_custom_lc(
        self,
        sector=None,
        tpf_size=None,
        sap_mask=None,
        aper_radius=None,
        percentile=None,
        threshold_sigma=None,
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
        pca_nterms : int
            number of pca terms to use

        Returns
        -------
        corrected_lc : lightkurve object
        """
        sector = sector if sector else self.sector
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

        aper_mask = parse_aperture_mask(
            tpf_tesscut,
            sap_mask=sap_mask,
            aper_radius=aper_radius,
            percentile=percentile,
            threshold_sigma=threshold_sigma,
            verbose=False,
        )
        self.aper_mask = aper_mask

        raw_lc = tpf_tesscut.to_lightcurve(
            method="aperture", aperture_mask=aper_mask
        )
        idx = (
            np.isnan(raw_lc.time)
            | np.isnan(raw_lc.flux)
            | np.isnan(raw_lc.flux_err)
        )
        raw_lc = raw_lc[~idx]
        self.lc_custom_raw = raw_lc
        # Make a design matrix and pass it to a linear regression corrector
        regressors = tpf_tesscut.flux[~idx][:, ~aper_mask]
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
        return lc

    def get_cdips_lc(
        self, sector=None, aper_idx=3, lctype="flux", verbose=False
    ):
        verbose = verbose if verbose is not None else self.verbose
        sector = sector if sector else self.sector
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
        self.lc_cdips = cdips.lc
        self.lc_cdips.targetid = self.ticid
        return cdips.lc


class ShortCadence(LongCadence):
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
        search_radius=3 * u.arcsec,
        apphot_method="sap",  # prf
        sap_mask="pipeline",
        quality_bitmask="default",
        apply_data_quality_mask=True,
        verbose=True,
        clobber=True,
        # mission="TESS",
        # quarter=None,
        # month=None,
        # campaign=None,
        # limit=None,
    ):
        super().__init__(
            sector=sector,
            name=name,
            toiid=toiid,
            ticid=ticid,
            epicid=epicid,
            gaiaDR2id=gaiaDR2id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            search_radius=search_radius,
            verbose=verbose,
            clobber=clobber,
            # sap_mask='threshold',
            # cutout_size = (50,50),
            # mission=mission
        )
        self.apphot_method = apphot_method
        self.sap_mask = sap_mask
        self.aper_mask = None
        self.quality_bitmask = quality_bitmask
        self.search_radius = search_radius
        self.data_quality_mask = None
        self.quality_bitmask = quality_bitmask
        self.apply_data_quality_mask = apply_data_quality_mask
        self.tpf = None
        self.lc_custom = None
        self.lc_custom_raw = None
        self.lcf = None
        self.lc_sap = None
        self.lc_pdcsap = None

    def get_lc(self, lctype="pdcsap", sector=None, quality_bitmask=None):
        """
        """
        sector = sector if sector else self.sector
        quality_bitmask = (
            quality_bitmask if quality_bitmask else self.quality_bitmask
        )
        self.all_sectors = self.get_all_sectors()
        if self.lcf is not None:
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
                assert q is not None, "Empty result. Check long cadence."
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
            if (sector == "all") & (len(self.all_sectors) > 1):
                lcf = q.download_all(quality_bitmask=quality_bitmask)
            else:
                lcf = q.download(quality_bitmask=quality_bitmask)
            self.lcf = lcf
        assert lcf is not None, "Empty result. Check long cadence."
        sap = lcf.SAP_FLUX
        pdcsap = lcf.PDCSAP_FLUX
        if isinstance(lcf, lk.LightCurveFileCollection):
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
            return pdcsap.remove_nans().normalize()
        else:
            return sap.remove_nans().normalize()

    def get_tpf(
        self,
        sector=None,
        apphot_method=None,
        quality_bitmask=None,
        return_df=True,
    ):
        """Download tpf from MAST given coordinates
           though using TIC id yields unique match.

        Parameters
        ----------
        sector : int
            TESS sector
        apphot_method : str
            aperture photometry method
        fitsoutdir : str
            fits output directory

        Returns
        -------
        tpf and/or df: lk.targetpixelfile, pd.DataFrame
        """
        # if self.verbose:
        #     print(f'Searching targetpixelfile using lightkurve')
        sector = sector if sector else self.sector
        apphot_method = apphot_method if apphot_method else self.apphot_method
        quality_bitmask = (
            quality_bitmask if quality_bitmask else self.quality_bitmask
        )
        if self.ticid:
            ticstr = f"TIC {self.ticid}"
            if self.verbose:
                print(f"\nSearching mast for {ticstr}\n")
            res = lk.search_targetpixelfile(
                ticstr, mission=MISSION, sector=None
            )
        else:
            if self.verbose:
                print(
                    f"\nSearching mast for ra,dec=({self.target_coord.to_string()})\n"
                )
            res = lk.search_targetpixelfile(
                self.target_coord,
                mission=MISSION,
                sector=None,  # search all if sector=None
            )
        df = res.table.to_pandas()

        if len(df) > 0:
            all_sectors = [int(i) for i in df["sequence_number"].values]
            if sector:
                sector_idx = df["sequence_number"][
                    df["sequence_number"].isin([sector])
                ].index.tolist()
                if len(sector_idx) == 0:
                    raise ValueError(
                        "sector {} data is unavailable".format(sector)
                    )
                obsid = df.iloc[sector_idx]["obs_id"].values[0]
                # ticid = int(df.iloc[sector_idx]["target_name"].values[0])
                fitsfilename = df.iloc[sector_idx]["productFilename"].values[0]
            else:
                sector_idx = 0
                sector = int(df.iloc[sector_idx]["sequence_number"])
                obsid = df.iloc[sector_idx]["obs_id"]
                # ticid = int(df.iloc[sector_idx]["target_name"])
                fitsfilename = df.iloc[sector_idx]["productFilename"]

            msg = f"{len(df)} tpf(s) found in sector(s) {all_sectors}\n"
            msg += f"Using data from sector {sector} only\n"
            if self.verbose:
                logging.info(msg)
                print(msg)

            filepath = join(
                fitsoutdir, "mastDownload/TESS", obsid, fitsfilename
            )
            if not exists(filepath) or self.clobber:
                if self.verbose:
                    print(f"Downloading TIC {self.ticid} ...\n")
                ticstr = f"TIC {self.ticid}"
                res = lk.search_targetpixelfile(
                    ticstr, mission=MISSION, sector=sector
                )
                tpf = res.download(
                    quality_bitmask=quality_bitmask, download_dir=fitsoutdir
                )
            else:
                if self.verbose:
                    print(
                        "Loading TIC {} from {}/...\n".format(
                            self.ticid, fitsoutdir
                        )
                    )
                tpf = lk.TessTargetPixelFile(filepath)
            if self.apply_data_quality_mask:
                tpf = remove_bad_data(tpf, sector=sector, verbose=self.verbose)
            self.tpf = tpf
            if return_df:
                return tpf, df
            else:
                return tpf
        else:
            msg = "No tpf file found! Check FFI data using --cadence=long\n"
            logging.info(msg)
            raise FileNotFoundError(msg)

    def get_aper_mask(
        self,
        sector=None,
        sap_mask=None,
        aper_radius=None,
        percentile=None,
        threshold_sigma=None,
        verbose=True,
    ):
        """
        """
        sector = sector if sector else self.sector
        sap_mask = sap_mask if sap_mask else self.sap_mask
        aper_radius = aper_radius if aper_radius else self.aper_radius
        percentile = percentile if percentile else self.percentile
        threshold_sigma = (
            threshold_sigma if threshold_sigma else self.threshold_sigma
        )

        if self.tpf is None:
            tpf, tpf_info = self.get_tpf(sector=sector, return_df=True)
        else:
            if self.tpf.sector != sector:
                tpf, tpf_info = self.get_tpf(sector=sector, return_df=True)
            else:
                tpf = self.tpf

        aper_mask = parse_aperture_mask(
            tpf,
            sap_mask=sap_mask,
            aper_radius=aper_radius,
            percentile=percentile,
            threshold_sigma=threshold_sigma,
            verbose=verbose,
        )
        self.aper_mask = aper_mask
        return aper_mask

    def make_custom_lc(
        self,
        sector=None,
        sap_mask=None,
        aper_radius=None,
        percentile=None,
        threshold_sigma=None,
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
        sector = sector if sector else self.sector
        sap_mask = sap_mask if sap_mask else self.sap_mask
        aper_radius = aper_radius if aper_radius else self.aper_radius
        percentile = percentile if percentile else self.percentile
        threshold_sigma = (
            threshold_sigma if threshold_sigma else self.threshold_sigma
        )

        tpf, tpf_info = self.get_tpf(sector=sector)
        # Make an aperture mask and a raw light curve
        aper_mask = parse_aperture_mask(
            tpf,
            sap_mask=sap_mask,
            aper_radius=aper_radius,
            percentile=percentile,
            threshold_sigma=threshold_sigma,
            verbose=False,
        )
        self.aper_mask = aper_mask

        raw_lc = tpf.to_lightcurve(aperture_mask=aper_mask)
        idx = (
            np.isnan(raw_lc.time)
            | np.isnan(raw_lc.flux)
            | np.isnan(raw_lc.flux_err)
        )
        raw_lc = raw_lc[~idx]
        self.lc_custom_raw = raw_lc
        # Make a design matrix and pass it to a linear regression corrector
        regressors = tpf.flux[~idx][:, ~aper_mask]
        self.lc_custom_raw = raw_lc
        dm = (
            lk.DesignMatrix(regressors, name="pixels")
            .pca(pca_nterms)
            .append_constant()
        )

        # Regression Corrector Object
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
        return lc


class LightCurve(ShortCadence, LongCadence):
    NotImplementedError


#     """
#     """
#     def __init__(
#         self,
#         cadence='short',
#         sap_mask='pipeline',
#         sector=None,
#         name=None,
#         toiid=None,
#         ticid=None,
#         epicid=None,
#         gaiaDR2id=None,
#         ra_deg=None,
#         dec_deg=None,
#         search_radius=3 * u.arcsec,
#         apphot_method="sap",  # prf
#         quality_bitmask="default",
#         cutout_size=(50, 50),
#         clobber=True,
#         verbose=True
#         ):
#         if cadence=='short':
#             super(ShortCadence, self).__init__(
#                 sector=sector,
#                 name=name,
#                 toiid=toiid,
#                 ticid=ticid,
#                 epicid=epicid,
#                 gaiaDR2id=gaiaDR2id,
#                 ra_deg=ra_deg,
#                 dec_deg=dec_deg,
#                 search_radius=search_radius,
#                 # quality_bitmask=quality_bitmask,
#                 verbose=verbose,
#                 clobber=clobber,
#                 )
#         else:
#             super(LongCadence, self).__init__(
#                 sector=sector,
#                 name=name,
#                 toiid=toiid,
#                 ticid=ticid,
#                 epicid=epicid,
#                 gaiaDR2id=gaiaDR2id,
#                 ra_deg=ra_deg,
#                 dec_deg=dec_deg,
#                 search_radius=search_radius,
#                 # quality_bitmask=quality_bitmask,
#                 sap_mask=sap_mask,
#                 cutout_size=cutout_size,
#                 clobber=clobber,
#                 verbose=verbose
#                 )
#         self.cadence=cadence

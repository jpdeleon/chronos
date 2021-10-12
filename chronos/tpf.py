# -*- coding: utf-8 -*-

r"""
Module for manipulating TESS targetpixelfile (tpf).
K2 tpf lives in k2 module.

NOTE: query_gaia_dr2_catalog method from Target hangs when called
so make sure to populate self.gaia_params first if needed
"""

# Import standard library
from time import time as timer
from os.path import join, exists
from requests.exceptions import HTTPError
import logging
import getpass

import numpy as np
import astropy.units as u
import lightkurve as lk

# Import from package
from chronos.config import DATA_PATH
from chronos.target import Target
from chronos.utils import (
    remove_bad_data,
    parse_aperture_mask,
    query_tpf,
    query_tpf_tesscut,
)
from chronos.constants import TESS_TIME_OFFSET

user = getpass.getuser()
MISSION = "TESS"
# TODO: ~/.astropy/cache/astroquery/*.pickle better default location than below?
fitsoutdir = join("/home", user, "data/transit")
log = logging.getLogger(__name__)

__all__ = ["Tpf", "FFI_cutout"]


class Tpf(Target):
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
        mission="tess",
        search_radius=3,
        sap_mask="pipeline",
        aper_radius=1,
        threshold_sigma=5,
        percentile=95,
        quality_bitmask="default",
        apply_data_quality_mask=False,
        clobber=True,
        verbose=True,
        calc_fpp=False,
        check_if_variable=False,
    ):
        """
        Attributes
        ----------
        calc_fpp : bool
            instantiates triceratops
        """
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
            check_if_variable=check_if_variable,
        )
        # self.mission = mission
        self.sector = sector
        # self.all_sectors = self.get_all_sectors()
        self.sap_mask = sap_mask
        self.aper_mask = None
        self.aper_radius = aper_radius
        self.percentile = percentile
        self.threshold_sigma = threshold_sigma
        self.quality_bitmask = quality_bitmask
        self.apply_data_quality_mask = apply_data_quality_mask
        self.tpf = None

        if calc_fpp:
            try:
                from triceratops import triceratops
            except Exception:
                errmsg = "pip install triceratops"
                raise ModuleNotFoundError(errmsg)
            try:
                self.triceratops = triceratops.target(
                    ID=self.ticid, sectors=self.all_sectors
                )
            except HTTPError:
                errmsg = "Check if target tpf is available in short cadence.\n"
                errmsg += "Short cadence targets only are currently supported."
                raise Exception(errmsg)
        self.calc_fpp = calc_fpp

        if self.sector is None:
            msg = "Target not found in any TESS sectors."
            assert len(self.all_sectors) > 0, msg
            self.sector = self.all_sectors[0]  # get first sector by default
        if self.sector == -1:
            self.sector = self.all_sectors[-1]
        print(f"Available sectors: {self.all_sectors}")
        print(f"Using sector={self.sector}.")

    def get_tpf(self, sector=None, quality_bitmask=None, return_df=False):
        """Download tpf from MAST given coordinates
           though using TIC id yields unique match.

        Parameters
        ----------
        sector : int
            TESS sector
        fitsoutdir : str
            fits output directory

        Returns
        -------
        tpf and/or df: lk.targetpixelfile, pd.DataFrame

        Note: find a way to compress the logic below
        if tpf is None:
            - download_tpf
        else:
            if tpf.sector==sector
                - load tpf
        else:
            - download_tpf
        """
        sector = sector if sector else self.sector
        quality_bitmask = (
            quality_bitmask if quality_bitmask else self.quality_bitmask
        )
        if self.tpf is None:
            if self.ticid is not None:
                # search by TICID
                ticstr = f"TIC {self.ticid}"
                if self.verbose:
                    print(f"\nSearching mast for {ticstr}.\n")
                res = lk.search_targetpixelfile(
                    ticstr, mission=MISSION, sector=None
                )
            else:
                # search by position
                if self.verbose:
                    print(
                        f"\nSearching mast for ra,dec=({self.target_coord.to_string()}).\n"
                    )
                res = lk.search_targetpixelfile(
                    self.target_coord,
                    mission=MISSION,
                    sector=None,  # search all if sector=None
                )
            assert res is not None, "No results from lightkurve search."
        else:
            # if self.tpf.sector == sector:
            #     # reload from memory
            #     tpf = self.tpf
            # else:
            if self.verbose:
                print("Searching targetpixelfile using lightkurve.")
            if self.ticid:
                ticstr = f"TIC {self.ticid}"
                if self.verbose:
                    print(f"\nSearching mast for {ticstr}.\n")
                res = lk.search_targetpixelfile(
                    ticstr, mission=MISSION, sector=None
                )
            else:
                if self.verbose:
                    print(
                        f"\nSearching mast for ra,dec=({self.target_coord.to_string()}).\n"
                    )
                res = lk.search_targetpixelfile(
                    self.target_coord,
                    mission=MISSION,
                    sector=None,  # search all if sector=None
                )
            assert res is not None, "No results from lightkurve search."
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

            msg = f"{len(df)} tpf(s) found in sector(s) {all_sectors}.\n"
            msg += f"Using data from sector {sector} only.\n"
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
                        "Loading TIC {} from {}/\n".format(
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
            if self.tpf.sector == sector:
                tpf = self.tpf
            else:
                tpf, tpf_info = self.get_tpf(sector=sector, return_df=True)

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

    def plot_field(self, mask=None, sector=None):
        """
        plot_field method of triceratops
        """
        errmsg = "Instantiate Tpf(calc_fpp=True)"
        assert self.calc_fpp, errmsg

        if mask is None:
            if self.tpf is not None:
                pix_xy = np.nonzero(self.tpf.pipeline_mask)
                # pixel locations of aperture
                aper = np.c_[
                    pix_xy[0] + self.tpf.column, pix_xy[1] + self.tpf.row
                ]
            else:
                raise ValueError("Use get_tpf().")
        else:
            aper = mask
        sector = self.sector if sector is None else sector
        self.triceratops.plot_field(sector=sector, ap_pixels=aper)
        # return fig

    def get_NEB_depths(self, mask=None, depth=None, recalc=False):
        """
        calc_depths method of triceratops
        """
        errmsg = "Instantiate Tpf(calc_fpp=True)"
        assert self.calc_fpp, errmsg

        if mask is None:
            if self.tpf is not None:
                mask = np.nonzero(self.tpf.pipeline_mask)
                # pixel locations of aperture
                aper = np.c_[mask[0] + self.tpf.column, mask[1] + self.tpf.row]
            else:
                raise ValueError("Use get_tpf().")
        else:
            aper = mask

        depth = self.toi_depth if depth is None else depth
        if (self.triceratops.stars is not None) or recalc:
            self.triceratops.calc_depths(tdepth=depth, all_ap_pixels=[aper])

        results = self.triceratops.stars.copy()
        return results

    def get_fpp(
        self,
        flat=None,
        fold=None,
        period=None,
        epoch=None,
        bin=None,
        cc_file=None,
        plot=True,
        recalc=False,
    ):
        """
        calc_probs method of triceratops
        See https://arxiv.org/abs/2002.00691
        Scenarios
        ---------
        Target has no unresolved stellar companion
        (bound or unbound) of significant flux
        * TTP: Target Transiting Planet
        * TEB: Target Eclipsing Binary
        ---------
        There's an unresolved bound stellar companion near the target star
        * PTP: Primary
        * PEB:
        * STP: Secondary
        * SEB:
        ---------
        There's an unresolved unbound stellar companion
        in the foreground or background along the line of sight
        to the target
        DTP: Diluted
        DEB:
        BTP: Background
        BEB:
        ---------
        Consequently,
        TFP: TEB+PEB+DEB (Target False Positive)
        CFP: STP+SEB+BTP+BEB (Companion FP)
        NFP: NTP+NEB (Nearby FP)
        triceratops is most effective at identifying
        small planet candidates that are actually CFP & NFP
        ---------
        Finally,
        FPP = 1 − (PTTP + PPTP + PDTP)
        """
        errmsg = "Instantiate Tpf(calc_fpp=True)"
        assert self.calc_fpp, errmsg

        period = self.toi_period if period is None else period
        epoch = self.toi_epoch if epoch is None else epoch
        if flat is not None:
            fold = flat.fold(period=period, t0=epoch - TESS_TIME_OFFSET)
        if fold is not None:
            fold = fold.remove_nans()
        else:
            errmsg = "Provide flat or fold lc."
            assert ValueError(errmsg)

        if bin is None:
            time, flux, flux_err = fold.time, fold.flux, fold.flux_err
        else:
            time, flux, flux_err = (
                fold.bin(bin).time,
                fold.bin(bin).flux,
                fold.bin(bin).flux_err,
            )

        if (not hasattr(self.triceratops, "probs")) or recalc:
            if not hasattr(self.triceratops.stars, "tdepth"):
                errmsg = "Try `self.get_NEB_depths()` first."
                raise ValueError(errmsg)
            if self.verbose:
                nstars = self.triceratops.stars.shape[0]
                # ETA is a wild guess here
                print(f"ETA: {nstars/10} mins.")
            time_start = timer()
            self.triceratops.calc_probs(
                time=time,
                flux_0=flux,
                flux_err_0=flux_err,
                P_orb=period,
                contrast_curve_file=cc_file,
            )
            hours, rem = divmod(timer() - time_start, 3600)
            minutes, seconds = divmod(rem, 60)
            if self.verbose:
                print(f"Run time: {int(minutes)}min {int(seconds)}sec.")
                print(f"FPP={self.triceratops.FPP:.4f}")
            errmsg = "Check fold lc for NaNs."
            assert not np.isnan(self.triceratops.FPP), errmsg
        if plot:
            self.triceratops.plot_fits(
                time=time, flux_0=flux, flux_err_0=flux_err, P_orb=period
            )
        results = self.triceratops.probs.copy()
        return results


class FFI_cutout(Target):
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
        mission="tess",
        search_radius=3,
        sap_mask="square",
        aper_radius=1,
        threshold_sigma=5,
        percentile=95,
        cutout_size=(15, 15),
        quality_bitmask="default",
        apply_data_quality_mask=False,
        calc_fpp=False,
        clobber=True,
        verbose=True,
        check_if_variable=False,
    ):
        """
        Attributes
        ----------
        calc_fpp : bool
            instantiates triceratops
        """
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
            check_if_variable=check_if_variable,
        )
        # self.mission = mission
        self.sector = sector
        # self.all_sectors = self.get_all_sectors()
        self.sap_mask = sap_mask
        self.aper_mask = None
        self.aper_radius = aper_radius
        self.percentile = percentile
        self.threshold_sigma = threshold_sigma
        self.cutout_size = cutout_size
        self.quality_bitmask = quality_bitmask
        self.apply_data_quality_mask = apply_data_quality_mask
        self.tpf_tesscut = None
        if calc_fpp:
            try:
                from triceratops import triceratops
            except Exception:
                errmsg = "pip install triceratops"
                raise ModuleNotFoundError(errmsg)
            try:
                self.triceratops = triceratops.target(
                    ID=self.ticid, sectors=self.all_sectors
                )
            except HTTPError:
                errmsg = "Check if target tpf is available in short cadence.\n"
                errmsg += "Short cadence targets only are currently supported."
                raise Exception(errmsg)
        self.calc_fpp = calc_fpp
        if self.sector is None:
            msg = "Target not found in any TESS sectors"
            assert len(self.all_sectors) > 0, msg
            self.sector = self.all_sectors[0]  # get first sector by default
        if self.sector == -1:
            self.sector = self.all_sectors[-1]
        print(f"Available sectors: {self.all_sectors}")
        print(f"Using sector={self.sector}.")

    def get_tpf_tesscut(self, sector=None, cutout_size=None):
        """
        """
        cutout_size = cutout_size if cutout_size else self.cutout_size
        sector = sector if sector else self.sector

        if self.tpf_tesscut is None:
            # download
            tpf = query_tpf_tesscut(
                self.target_coord,
                sector=sector,
                quality_bitmask=self.quality_bitmask,
                verbose=self.verbose,
                cutout_size=cutout_size,
                apply_data_quality_mask=self.apply_data_quality_mask,
            )
        else:
            if (self.tpf_tesscut.sector == sector) & (
                self.cutout_size == cutout_size
            ):
                # reload from memory
                tpf = self.tpf_tesscut
            else:
                # download
                tpf = query_tpf_tesscut(
                    self.target_coord,
                    sector=sector,
                    quality_bitmask=self.quality_bitmask,
                    verbose=self.verbose,
                    cutout_size=cutout_size,
                    apply_data_quality_mask=self.apply_data_quality_mask,
                )
        if self.ticid is not None:
            tpf.targetid = self.ticid
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

        tpf = self.get_tpf_tesscut(sector=sector, cutout_size=cutout_size)

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

    def plot_field(self, mask, sector=None):
        """
        plot_field function of triceratops
        """
        errmsg = "Instantiate FFI_cutout(calc_fpp=True)"
        assert self.calc_fpp, errmsg
        # pixel locations of aperture
        pix_xy = np.nonzero(mask)
        # pixel locations of aperture
        aper = np.c_[
            pix_xy[0] + self.tpf_tesscut.column,
            pix_xy[1] + self.tpf_tesscut.row,
        ]
        sector = self.sector if sector is None else sector
        self.triceratops.plot_field(sector=sector, ap_pixels=aper)
        # return fig

    def get_NEB_depths(self, mask, depth=None):
        """
        TODO: use multiple aper
        """
        errmsg = "Instantiate FFI_cutout(calc_fpp=True)"
        assert self.calc_fpp, errmsg
        # pixel locations of aperture
        pix_xy = np.nonzero(mask)
        # pixel locations of aperture
        aper = np.c_[
            pix_xy[0] + self.tpf_tesscut.column,
            pix_xy[1] + self.tpf_tesscut.row,
        ]
        depth = self.toi_depth if depth is None else depth
        self.triceratops.calc_depths(tdepth=depth, all_ap_pixels=[aper])
        return self.triceratops.stars.copy()

    def get_fpp(
        self,
        flat=None,
        fold=None,
        bin=None,
        period=None,
        epoch=None,
        cc_file=None,
        plot=True,
        recalc=False,
    ):
        """
        calc_probs method of triceratops
        See https://arxiv.org/abs/2002.00691
        Scenarios
        ---------
        Target has no unresolved stellar companion
        (bound or unbound) of significant flux
        * TTP: Target Transiting Planet
        * TEB: Target Eclipsing Binary
        ---------
        There's an unresolved bound stellar companion near the target star
        * PTP: Primary
        * PEB:
        * STP: Secondary
        * SEB:
        ---------
        There's an unresolved unbound stellar companion
        in the foreground or background along the line of sight
        to the target
        DTP: Diluted
        DEB:
        BTP: Background
        BEB:
        ---------
        Consequently,
        TFP: TEB+PEB+DEB (Target False Positive)
        CFP: STP+SEB+BTP+BEB (Companion FP)
        NFP: NTP+NEB (Nearby FP)
        triceratops is most effective at identifying
        small planet candidates that are actually CFP & NFP
        ---------
        Finally,
        FPP = 1 − (PTTP + PPTP + PDTP)
        """
        errmsg = "Instantiate Tpf(calc_fpp=True)"
        assert self.calc_fpp, errmsg

        period = self.toi_period if period is None else period
        epoch = self.toi_epoch if epoch is None else epoch
        if flat is not None:
            fold = flat.fold(period=period, t0=epoch - TESS_TIME_OFFSET)
        if fold is not None:
            fold = fold.remove_nans()
        else:
            errmsg = "Provide flat or fold lc."
            assert ValueError(errmsg)

        if bin is None:
            time, flux, flux_err = fold.time, fold.flux, fold.flux_err
        else:
            time, flux, flux_err = (
                fold.bin(bin).time,
                fold.bin(bin).flux,
                fold.bin(bin).flux_err,
            )
        if (not hasattr(self.triceratops, "probs")) or recalc:
            if not hasattr(self.triceratops.stars, "tdepth"):
                errmsg = "Try `self.get_NEB_depths()` first."
                raise ValueError(errmsg)
            if self.verbose:
                nstars = self.triceratops.stars.shape[0]
                # ETA is a wild guess here
                print(f"ETA: {nstars/10} mins.")
            time_start = timer()
            self.triceratops.calc_probs(
                time=time,
                flux_0=flux,
                flux_err_0=flux_err,
                P_orb=period,
                # contrast_curve_file=cc_file,
            )
            hours, rem = divmod(timer() - time_start, 3600)
            minutes, seconds = divmod(rem, 60)
            fpp = self.triceratops.FPP
            if self.verbose:
                print(f"Run time: {int(minutes)}min {int(seconds)}sec.")
                print(f"FPP={fpp:.4f}")
            errmsg = "Check fold lc for NaNs."
            assert not np.isnan(fpp), errmsg
        if plot:
            self.triceratops.plot_fits(
                time=time, flux_0=flux, flux_err_0=flux_err, P_orb=period
            )
        results = self.triceratops.probs.copy()
        return results

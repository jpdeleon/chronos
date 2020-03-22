# -*- coding: utf-8 -*-

r"""
classes for manipulating targetpixelfile
"""

# Import standard library
from os.path import join, exists
import logging
import getpass

# import numpy as np
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

user = getpass.getuser()
MISSION = "TESS"
fitsoutdir = join("/home", user, "data/transit")
log = logging.getLogger(__name__)

__all__ = ["Tpf", "Tpf_cutout"]


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
        search_radius=3 * u.arcsec,
        sap_mask="pipeline",
        aper_radius=1,
        threshold_sigma=5,
        percentile=95,
        quality_bitmask="default",
        apply_data_quality_mask=False,
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
        self.all_sectors = self.get_all_sectors()
        self.sap_mask = sap_mask
        self.aper_mask = None
        self.aper_radius = aper_radius
        self.percentile = percentile
        self.threshold_sigma = threshold_sigma
        self.search_radius = search_radius
        self.quality_bitmask = quality_bitmask
        self.apply_data_quality_mask = apply_data_quality_mask
        self.tpf = None

        if self.sector is None:
            self.sector = self.all_sectors[0]  # get first sector by default
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
        if self.verbose:
            print(f"Searching targetpixelfile using lightkurve")
        sector = sector if sector else self.sector
        quality_bitmask = (
            quality_bitmask if quality_bitmask else self.quality_bitmask
        )
        if self.tpf is None:
            if self.ticid is not None:
                # search by TICID
                ticstr = f"TIC {self.ticid}"
                if self.verbose:
                    print(f"\nSearching mast for {ticstr}\n")
                res = lk.search_targetpixelfile(
                    ticstr, mission=MISSION, sector=None
                )
            else:
                # search by position
                if self.verbose:
                    print(
                        f"\nSearching mast for ra,dec=({self.target_coord.to_string()})\n"
                    )
                res = lk.search_targetpixelfile(
                    self.target_coord,
                    mission=MISSION,
                    sector=None,  # search all if sector=None
                )
        else:
            if self.tpf.sector == sector:
                # reload from memory
                tpf = self.tpf
            else:
                if self.verbose:
                    print("Searching targetpixelfile using lightkurve")
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


class Tpf_cutout(Target):
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
        sap_mask="square",
        aper_radius=1,
        threshold_sigma=5,
        percentile=95,
        cutout_size=(15, 15),
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
        self.all_sectors = self.get_all_sectors()
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

        if self.sector is None:
            msg = f"Target not found in any TESS sectors"
            assert len(self.all_sectors) > 0, msg
            self.sector = self.all_sectors[0]  # get first sector by default
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

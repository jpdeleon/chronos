# -*- coding: utf-8 -*-

r"""
Module for star bookkeeping, e.g. position, catalog cross-matching, archival data look-up.
"""

# Import standard library
# from inspect import signature
from pathlib import Path
import warnings
from pprint import pprint
import logging

# Import modules
# from matplotlib.figure import Figure
# from matplotlib.image import AxesImage
# from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.mast import Observations, Catalogs
from astropy.coordinates import SkyCoord, Distance
import astropy.units as u
import lightkurve as lk
from tqdm import tqdm

# Import from package
from chronos.config import DATA_PATH
from chronos.cluster import ClusterCatalog, Cluster
from chronos.gls import Gls
from chronos.utils import (
    get_all_campaigns,
    get_all_sectors,
    get_tess_ccd_info,
    get_toi,
    get_tois,
    get_target_coord,
    get_target_coord_3d,
    get_epicid_from_k2name,
    get_harps_bank,
    get_specs_table_from_tfop,
    get_between_limits,
    get_above_lower_limit,
    get_below_upper_limit,
    flatten_list,
    get_TGv8_catalog,
    get_k2_data_from_exofop,
)

log = logging.getLogger(__name__)

__all__ = ["Target"]


class Target(object):
    """
    Performs target resolution basic catalog cross-matching and archival data look-up
    """

    def __init__(
        self,
        name=None,
        toiid=None,
        ctoiid=None,
        ticid=None,
        epicid=None,
        gaiaDR2id=None,
        ra_deg=None,
        dec_deg=None,
        mission="tess",
        search_radius=3,
        verbose=True,
        clobber=False,
        check_if_variable=False,
    ):
        """
        Attributes
        ----------
        search_radius : float
            search radius for matching [arcsec]
        """
        self.clobber = clobber
        self.verbose = verbose
        self.mission = mission.lower()
        self.toi_params = None
        self.nea_params = None
        self.toiid = toiid if toiid is None else int(toiid)  # e.g. 837
        self.ctoiid = ctoiid  # e.g. 364107753.01
        self.ticid = ticid if ticid is None else int(ticid)  # e.g. 364107753
        self.epicid = epicid if epicid is None else int(epicid)  # 201270176
        self.gaiaid = gaiaDR2id  # e.g. Gaia DR2 5251470948229949568
        self.ra = ra_deg
        self.dec = dec_deg
        self.target_name = name  # e.g. Pi Mensae
        # determine target name
        if self.toiid is not None:
            name = f"TOI {self.toiid}"
        elif self.ticid is not None:
            name = f"TIC {self.ticid}"
        elif self.epicid is not None:
            name = f"EPIC {self.epicid}"
            self.mission = "k2"
        elif self.gaiaid is not None:
            name = f"Gaia DR2 {self.gaiaid}"
        elif self.target_name is not None:
            if self.target_name[:2].lower() == "k2":
                name = self.target_name.upper()
                self.mission = "k2"
                self.epicid = get_epicid_from_k2name(name)
            elif self.target_name[:6].lower() == "kepler":
                name = self.target_name.title()
                self.mission = "kepler"
            elif self.target_name[:4].lower() == "gaia":
                name = self.target_name.upper()
                if gaiaDR2id is None:
                    self.gaiaid = int(name.strip()[4:])
        elif (self.ra is not None) & (self.dec is not None):
            name = f"({self.ra:.3f}, {self.dec:.3f})"
        # specify name
        if self.target_name is None:
            self.target_name = name
        # check if TIC is a TOI
        if (self.ticid is not None) and (self.toiid is None):
            tois = get_tois(clobber=True, verbose=False)
            idx = tois["TIC ID"].isin([self.ticid])
            if sum(idx) > 0:
                self.toiid = tois.loc[idx, "TOI"].values[0]
                if self.verbose:
                    print(f"TIC {self.ticid} is TOI {int(self.toiid)}!")
        # query TOI params
        if self.toiid is not None:
            self.toi_params = get_toi(
                toi=self.toiid, clobber=self.clobber, verbose=False
            ).iloc[0]
            # nplanets = int(self.toi_params["Planet Num"])
            # if nplanets > 1:
            #     print(f"Target has {nplanets} planets.")
        if (self.ticid is None) and (self.toiid is not None):
            self.ticid = int(self.toi_params["TIC ID"])
        # get coordinates
        self.target_coord = get_target_coord(
            ra=self.ra,
            dec=self.dec,
            toi=self.toiid,
            ctoi=self.ctoiid,
            tic=self.ticid,
            epic=self.epicid,
            gaiaid=self.gaiaid,
            name=self.target_name,
        )
        if self.mission == "tess":
            self.all_sectors = get_all_sectors(self.target_coord)
            self.tess_ccd_info = get_tess_ccd_info(self.target_coord)
        elif (self.mission == "k2") | (self.mission == "kepler"):
            try:
                self.all_campaigns = get_all_campaigns(self.epicid)
            except Exception:
                # error when GaiaDR2id is only given
                print("mission=Kepler/K2 but no epicid given.")
            # this does not throw an error
            # self.tess_ccd_info = tesscut.Tesscut.get_sectors(self.target_coord).to_pandas()
            # if self.tess_ccd_info is not None:
            #     self.all_sectors = np.array([int(i) for i in self.tess_ccd_info["sector"].values])
            #     tic_params = self.query_tic_catalog(return_nearest_xmatch=True)
            #     print(f"Target is also TIC {tic_params['ID']}")

        self.search_radius = search_radius * u.arcsec
        self.tic_params = None
        self.gaia_params = None
        self.gaia_sources = None
        self.gmag = None
        self.distance_to_nearest_cluster_member = None
        self.nearest_cluster_member = None
        self.nearest_cluster_members = None
        self.nearest_cluster_name = None
        self.vizier_tables = None
        self.cc = None
        # as opposed to self.cc.all_clusters, all_clusters has uncertainties
        # appended in get_cluster_membership
        self.all_clusters = None
        self.harps_bank_table = None
        self.harps_bank_rv = None
        self.harps_bank_target_name = None
        self.variable_star = False
        if self.verbose:
            print(f"Target: {name}")
        if check_if_variable:
            self.query_variable_star_catalogs()

    def __repr__(self):
        """Override to print a readable string representation of class
        """
        # params = signature(self.__init__).parameters
        # val = repr(getattr(self, key))

        included_args = [
            # ===target attributes===
            "name",
            "toiid",
            "ctoiid",
            "ticid",
            "epicid",
            "gaiaDR2id",
            "ra_deg",
            "dec_deg",
            "target_coord",
            "search_radius",
            "mission",
            "campaign",
            "all_sectors",
            "all_campaigns",
            # ===tpf attributes===
            "sap_mask",
            "quality_bitmask",
            "calc_fpp",
            # 'aper_radius', 'threshold_sigma', 'percentile' #if sap_mask!='pipeline'
            # cutout_size #for FFI
            # ===lightcurve===
            "lctype",
            "aper_idx",
        ]
        args = []
        for key in self.__dict__.keys():
            val = self.__dict__.get(key)
            if key in included_args:
                if key == "target_coord":
                    # format coord
                    coord = self.target_coord.to_string("decimal")
                    args.append(f"{key}=({coord.replace(' ',',')})")
                elif val is not None:
                    args.append(f"{key}={val}")
        args = ", ".join(args)
        return f"{type(self).__name__}({args})"

    # def __repr__(self):
    #     params = signature(self.__init__).parameters
    #     values = (f"{p}={repr(getattr(self, p))}" for p in params)
    #     return f"{type(self).__name__}({', '.join(values)})"

    def query_variable_star_catalogs(self):
        """
        Check for variable star flag in vizier and var in catalog title
        """
        # tabs = self.query_vizier_param('var')
        # if len(tabs)>1:
        #     print(tabs)
        #     print("***Target has a variable star flag!***")
        #     self.variable_star = True
        all_tabs = self.query_vizier(verbose=False)
        # check for `var` in catalog title
        idx = [
            n if "var" in t._meta["description"] else False
            for n, t in enumerate(all_tabs)
        ]
        for i in idx:
            if i:
                tab = all_tabs[i]
                s = tab.to_pandas().squeeze().str.decode("ascii")
                print(f"\nSee also: {tab._meta['name']}\n{s}")
                self.variable_star = True

    def query_M_dwarf_catalog(self):
        """
        http://spider.ipac.caltech.edu/staff/davy/ARCHIVE/index.shtml
        """
        return NotImplementedError("method to be added soon")

    def query_hypatia_catalog(self):
        """
        stellar abundance data
        https://www.hypatiacatalog.com/
        """
        return NotImplementedError("method to be added soon")

    def query_TGv8_catalog(self, gaiaid=None, data_path=None):
        """
        Stellar parameters of TESS host stars (TICv8)
        using Gaia2+APOGEE14+GALAH+RAVE5+LAMOST+SkyMapper;
        See Carillo2020: https://arxiv.org/abs/1911.07825
        Parameter
        ---------
        gaiaid : int
            Gaia DR2 source id (optional)
        data_path : str
            path to data

        Returns
        -------
        pandas.Series
        """
        if gaiaid is None:
            if self.gaiaid is None:
                errmsg = "Provide gaiaid or try `self.query_gaia_dr2_catalog"
                errmsg += "(return_nearest_xmatch=True)`"
                raise ValueError(errmsg)
            else:
                gaiaid = self.gaiaid

        df = get_TGv8_catalog(data_path=data_path)
        d = df.query("Gaia_source_id==@gaiaid")
        if len(d) > 0:
            # return series
            return d.squeeze()
        else:
            print(f"Gaia DR2 {gaiaid} not found in TGv8 catalog.")

    def query_gaia_dr2_catalog(
        self, radius=None, return_nearest_xmatch=False, verbose=None
    ):
        """
        cross-match to Gaia DR2 catalog by angular separation
        position (accounting for proper motion) and brightess
        (comparing Tmag to Gmag whenever possible)

        Take caution:
        * phot_proc_mode=0 (i.e. “Gold” sources, see Riello et al. 2018)
        * astrometric_excess_noise_sig < 5
        * astrometric_gof_al < 20
        * astrometric_chi2_al
        * astrometric_n_good_obs_al
        * astrometric_primary_flag
        * duplicated source=0
        * visibility_periods_used
        * phot_variable_flag
        * flame_flags
        * priam_flags
        * phot_(flux)_excess_factor
        (a measure of the inconsistency between GBP, G, and GRP bands
        typically arising from binarity, crowdening and incomplete background
        modelling).

        Parameter
        ---------
        radius : float
            query radius in arcsec
        return_nearest_xmatch : bool
            return nearest single star if True else possibly more matches

        Returns
        -------
        tab : pandas.DataFrame
            table of star match(es)

        Notes:
        1. See column meaning here: https://mast.stsci.edu/api/v0/_c_a_o_mfields.html

        2. Gaia DR2 parallax has -0.08 mas offset (Stassun & Toress 2018,
        https://arxiv.org/pdf/1805.03526.pdf)

        3. quadratically add 0.1 mas to the uncertainty to account for systematics
        in the Gaia DR2 data (Luri+2018)

        4. Gmag has an uncertainty of 0.01 mag (Casagrande & VandenBerg 2018)

        From Carillo+2019:
        The sample with the low parallax errors i.e. 0 < f < 0.1,
        has distances derived from simply inverting the parallax

        Whereas, the sample with higher parallax errors i.e. f > 0.1
        has distances derived from a Bayesian analysis following Bailer-Jones (2015),
        where they use a weak distance prior (i.e. exponentially decreasing space
        density prior) that changes with Galactic latitude and longitude

        5. See also Gaia DR2 Cross-match for the celestial reference system (ICRS)
        https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_proc/ssec_cu3ast_proc_xmatch.html
        and
        https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_cali/ssec_cu3ast_cali_frame.html

        6. See https://github.com/tzdwi/TESS-Gaia and https://github.com/JohannesBuchner/nway
        and Salvato+2018 Appendix A for catalog matching problem: https://arxiv.org/pdf/1705.10711.pdf

        See also CDIPS gaia query:
        https://github.com/lgbouma/cdips/blob/master/cdips/utils/gaiaqueries.py

        See also bulk query:
        https://gea.esac.esa.int/archive-help/tutorials/python_cluster/index.html
        """
        radius = self.search_radius if radius is None else radius * u.arcsec
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            # silenced when verbose=False instead of None
            print(
                f"""Querying Gaia DR2 catalog for ra,dec=({self.target_coord.to_string()}) within {radius}."""
            )
        # load gaia params for all TOIs
        tab = Catalogs.query_region(
            self.target_coord, radius=radius, catalog="Gaia", version=2
        ).to_pandas()
        # rename distance to separation because it is confusing
        tab = tab.rename(columns={"distance": "separation"})
        # convert from arcmin to arcsec
        tab["separation"] = tab["separation"].apply(
            lambda x: x * u.arcmin.to(u.arcsec)
        )
        errmsg = f"No gaia star within {radius}. Use radius>{radius}"
        assert len(tab) > 0, errmsg
        tab["source_id"] = tab.source_id.astype(int)
        # check if results from DR2 (epoch 2015.5)
        assert np.all(
            tab["ref_epoch"].isin([2015.5])
        ), "Epoch not 2015 (version<2?)"

        if return_nearest_xmatch:
            nearest_match = tab.iloc[0]
            tplx = float(nearest_match["parallax"])
            if np.isnan(tplx) | (tplx < 0):
                print(f"Target parallax ({tplx} mas) is omitted!")
                tab["parallax"] = np.nan
        else:
            nstars = len(tab)
            idx1 = tab["parallax"] < 0
            tab.loc[idx1, "parallax"] = np.nan  # replace negative with nan
            idx2 = tab["parallax"].isnull()
            errmsg = f"No stars within radius={radius} have positive Gaia parallax!\n"
            if idx1.sum() > 0:
                errmsg += (
                    f"{idx1.sum()}/{nstars} stars have negative parallax!\n"
                )
            if idx2.sum() > 0:
                errmsg += f"{idx2.sum()}/{nstars} stars have no parallax!"
            assert len(tab) > 0, errmsg
        """
        FIXME: check parallax error here and apply corresponding distance calculation: see Note 1
        """
        if self.gaiaid is not None:
            errmsg = "Catalog does not contain target gaia id."
            assert np.any(tab["source_id"].isin([self.gaiaid])), errmsg

        # add gaia distance to target_coord
        # FIXME: https://docs.astropy.org/en/stable/coordinates/transforming.html
        gcoords = SkyCoord(
            ra=tab["ra"],
            dec=tab["dec"],
            unit="deg",
            frame="icrs",
            obstime="J2015.5",
        )
        # precess coordinate from Gaia DR2 epoch to J2000
        gcoords = gcoords.transform_to("icrs")
        if self.gaiaid is None:
            # find by nearest distance (for toiid or ticid input)
            idx = self.target_coord.separation(gcoords).argmin()
        else:
            # find by id match for gaiaDR2id input
            idx = tab.source_id.isin([self.gaiaid]).argmax()
        star = tab.loc[idx]
        # get distance from parallax
        if star["parallax"] > 0:
            target_dist = Distance(parallax=star["parallax"] * u.mas)
        else:
            target_dist = np.nan

        # redefine skycoord with coord and distance
        target_coord = SkyCoord(
            ra=self.target_coord.ra,
            dec=self.target_coord.dec,
            distance=target_dist,
        )
        self.target_coord = target_coord

        nsources = len(tab)
        if return_nearest_xmatch or (nsources == 1):
            if nsources > 1:
                print(f"There are {nsources} gaia sources within {radius}.")
            target = tab.iloc[0]
            if self.gaiaid is not None:
                id = int(target["source_id"])
                msg = f"Nearest match ({id}) != {self.gaiaid}"
                assert int(self.gaiaid) == id, msg
            else:
                self.gaiaid = int(target["source_id"])
            self.gaia_params = target
            self.gmag = target["phot_g_mean_mag"]
            ens = target["astrometric_excess_noise_sig"]
            if ens >= 5:
                msg = f"astrometric_excess_noise_sig>{ens:.2f} (>5 hints binarity).\n"
                print(msg)
            gof = target["astrometric_gof_al"]
            if gof >= 20:
                msg = f"astrometric_gof_al>{gof:.2f} (>20 hints binarity)."
                print(msg)
            if (ens >= 5) or (gof >= 20):
                print("See https://arxiv.org/pdf/1804.11082.pdf\n")
            delta = np.hypot(target["pmra"], target["pmdec"])
            if abs(delta) > 10:
                print("High proper-motion star:")
                print(
                    f"(pmra,pmdec)=({target['pmra']:.2f},{target['pmdec']:.2f}) mas/yr"
                )
            if target["visibility_periods_used"] < 6:
                msg = "visibility_periods_used<6 so no astrometric solution\n"
                msg += "See https://arxiv.org/pdf/1804.09378.pdf\n"
                print(msg)
            return target  # return series of len 1
        else:
            # if self.verbose:
            #     d = self.get_nearby_gaia_sources()
            #     print(d)
            self.gaia_sources = tab
            return tab  # return dataframe of len 2 or more

    def query_tic_catalog(self, radius=None, return_nearest_xmatch=False):
        """
        Query TIC v8 catalog from MAST: https://astroquery.readthedocs.io/en/latest/mast/mast.html
        See column meaning in https://mast.stsci.edu/api/v0/_t_i_cfields.html
        and Table B in Stassun+2019: https://arxiv.org/pdf/1905.10694.pdf

        Parameter
        ---------
        radius : float
            query radius in arcsec

        Returns
        -------
        tab : pandas.DataFrame
            table of star match(es)
        """
        radius = self.search_radius if radius is None else radius * u.arcsec
        if self.verbose:
            print(
                f"Querying TIC catalog for ra,dec=({self.target_coord.to_string()}) within {radius}."
            )
        # NOTE: check tic version
        tab = Catalogs.query_region(
            self.target_coord, radius=radius, catalog="TIC"
        ).to_pandas()
        errmsg = f"No TIC star within {self.search_radius}"
        nsources = len(tab)
        assert nsources > 0, errmsg
        if return_nearest_xmatch or (nsources == 1):
            if nsources > 1:
                print(f"There are {nsources} TIC stars within {radius}")
            # get nearest match
            tab = tab.iloc[0]
            if tab.wdflag == 1:
                print("white dwarf flag = True!")
            if self.ticid is not None:
                id = int(tab["ID"])
                msg = f"Nearest match ({id}) != {self.ticid}"
                assert int(self.ticid) == id, msg
            else:
                if self.ticid is None:
                    self.ticid = int(tab["ID"])
        self.tic_params = tab
        return tab

    def validate_gaia_tic_xmatch(self, Rtol=0.3, mtol=0.5):
        """
        check if Rstar and parallax from 2 catalogs match,
        raises error otherwise
        """
        if (self.gaia_params is None) or (
            isinstance(self.gaia_params, pd.DataFrame)
        ):
            msg = "run query_gaia_dr2_catalog(return_nearest_xmatch=True)"
            raise ValueError(msg)
        g = self.gaia_params
        if (self.tic_params is None) or (
            isinstance(self.tic_params, pd.DataFrame)
        ):
            msg = "run query_tic_catalog(return_nearest_xmatch=True)"
            raise ValueError(msg)
        t = self.tic_params

        # check magnitude
        if np.any(np.isnan([g.phot_g_mean_mag, t.Tmag])):
            msg = f"Gmag={g.phot_g_mean_mag}; Tmag={t.Tmag}"
            warnings.warn(msg)
            print(msg)
        else:
            assert np.allclose(g.phot_g_mean_mag, t.Tmag, rtol=mtol)

        # check parallax

        if np.any(np.isnan([g.parallax, t.plx])):
            msg = f"Gaia parallax={g.parallax}; TIC parallax={t.plx}"
            warnings.warn(msg)
            print(msg)
        else:
            assert np.allclose(g.parallax, t.plx, rtol=1e-3)

        # check Rstar
        if np.any(np.isnan([g.radius_val, t.rad])):
            msg = f"Gaia radius={g.radius_val}; TIC radius={t.rad}"
            warnings.warn(msg)
            print(msg)
        else:
            # dradius = g.radius_val - t.rad
            # msg = f"Rgaia-Rtic={g.radius_val:.2f}-{t.rad:.2f}={dradius:.2f}"
            # assert dradius <= Rtol, msg
            assert np.allclose(g.radius_val, t.rad, rtol=Rtol)

        # check gaia ID
        if (self.gaiaid is not None) and (t["GAIA"] is not np.nan):
            assert g.source_id == int(t["GAIA"]), "Different source IDs!"

        print("Gaia and TIC catalog cross-match succeeded.")
        return True

    def validate_gaia_epic_xmatch(self, mtol=0.5):
        """
        """
        errmsg = "Under development"
        raise NotImplementedError(errmsg)
        # if (self.epicid is not None) & (self.mission == "k2"):
        #     # query vizier parameters of nearest cross-match
        #     if self.vizier_tables is None:
        #         tab = self.query_vizier()
        #     else:
        #         tab = self.vizier_tables
        # e = tab["J/ApJS/224/2/table5"].to_pandas().squeeze()  # epic catalog
        # g = tab["I/345/gaia2"].to_pandas().squeeze()  # gaia dr2 catalog
        # # check magnitude
        # if np.any(np.isnan([g.Gmag, e.Kepmag])):
        #     msg = f"Gmag={g.Gmag}; Kepmag={t.Kepmag}"
        #     warnings.warn(msg)
        #     print(msg)
        # else:
        #     assert np.allclose(g.Gmag, t.Kepmag, rtol=mtol)
        # errmsg = "Different EPIC IDs!"
        # assert e.EPIC == self.epicid, errmsg
        # print("Gaia and TIC catalog cross-match succeeded.")
        # return True

    def get_nearby_gaia_sources(self, radius=60, depth=None, add_column=None):
        """
        get information about stars within radius [arcsec] and
        dilution factor from delta Gmag

        Parameters
        ----------
        radius : float
            query radius in arcsec
        add_column : str
            additional Gaia column name to show (e.g. radial_velocity)
        """
        radius = radius if radius is not None else 60

        if (self.gaia_sources is not None) & (radius == 60):
            d = self.gaia_sources.copy(deep=True)
        else:
            d = self.query_gaia_dr2_catalog(radius=radius).copy(deep=True)

        if self.gaiaid is None:
            # nearest match (first entry row=0) is assumed as the target
            gaiaid = int(d.iloc[0]["source_id"])
        else:
            gaiaid = self.gaiaid
        msg = f"Only 1 gaia source found within r={radius} arcsec"
        assert isinstance(d, pd.DataFrame), msg
        idx = d.source_id == gaiaid
        target_gmag = d.loc[idx, "phot_g_mean_mag"].values[0]
        d["delta_Gmag"] = d["phot_g_mean_mag"] - target_gmag
        # compute dilution factor
        d["gamma_pri"] = 1 + 10 ** (-0.4 * d["delta_Gmag"])
        d["gamma_sec"] = 1 + 10 ** (0.4 * d["delta_Gmag"])
        columns = [
            "source_id",
            "parallax",
            "astrometric_gof_al",
            "astrometric_excess_noise_sig",
            "separation",
            "phot_g_mean_mag",
            "delta_Gmag",
            "gamma_pri",
            "gamma_sec",
        ]
        if depth is None:
            if self.toi_depth is not None:
                depth = self.toi_depth
            else:
                print("Supply depth, else depth=0.")
                depth = 0
        if depth is not None:
            d["true_depth_pri"] = d["gamma_pri"] * depth
            d["true_depth_sec"] = d["gamma_sec"] * depth
            columns.append("true_depth_pri")
            columns.append("true_depth_sec")
            col = "true_depth_sec>1(cleared?)"
            columns.append(col)
            d[col] = d.true_depth_sec > 1

        if add_column is not None:
            assert (isinstance(add_column, str)) & (add_column in d.columns)
            columns.append(add_column)
        return d[columns]

    def get_max_Tmag_from_depth(self, depth=None):
        """
        depth : float
            TESS transit depth
        """
        if depth is None:
            if self.toi_depth is not None:
                depth = self.toi_depth
            else:
                print("Supply depth, else depth=0.")
                depth = 0

        if self.tic_params is None:
            tic_params = self.query_tic_catalog(return_nearest_xmatch=True)
        else:
            tic_params = self.tic_params
        Tmag = tic_params["Tmag"]
        dT = -2.5 * np.log10(depth)
        Tmax = Tmag + dT
        if self.verbose:
            print(
                f"Given depth={depth*100:.4f}%, Tmag={Tmax:.2f} is the max. mag of a resolved companion that can reproduce this transit."
            )
        return Tmax

    def get_possible_NEBs(self, depth, gaiaid=None, kmax=1.0):
        """
        depth is useful to rule out deep eclipses when depth*gamma > kmax

        kmax : float [0,1]
            maximum eclipse depth (default=1)
        """
        assert (kmax >= 0.0) & (kmax <= 1.0), "eclipse depth is between 0 & 1"

        d = self.get_nearby_gaia_sources()

        good, bad = [], []
        for _, row in d.iterrows():
            gaiaid, dmag, gamma = row[["source_id", "delta_Gmag", "gamma"]]
            if int(gaiaid) != gaiaid:
                if depth * gamma > kmax:
                    # observed depth is too deep to have originated from the secondary star
                    good.append(gaiaid)
                else:
                    # uncertain signal source
                    bad.append(gaiaid)
        uncleared = d.loc[d.source_id.isin(bad)]
        return uncleared

    def get_cluster_membership(
        self,
        catalog_name="CantatGaudin2020",
        frac_err=0.1,
        sigma=5,
        return_idxs=False,
        verbose=None,
    ):
        """
        Check vizier if target is known as a cluster/assoc member.
        Find the cluster that matches the 6D kinematics of the target
        Parameters
        ----------
        catalog_name : str
            cluster catalog to search for
        frac_err : float
            minimum fractional error of the parameter;
            parameters greater than frac_err are ignored
        sigma : float
            minimum sigma of a parameter;
            parameters greater than `sigma` away from
            the cluster parameter distribution are not member
        return_idxs : bool
            return indexes if True
        """
        verbose = verbose if verbose is not None else self.verbose
        if self.gaia_params is None:
            gaia_params = self.query_gaia_dr2_catalog(
                return_nearest_xmatch=True
            )
        else:
            gaia_params = self.gaia_params
        params = "ra dec parallax pmra pmdec RV".split()
        gparams = "ra dec parallax pmra pmdec radial_velocity".split()

        if self.cc is None:
            self.cc = ClusterCatalog(catalog_name=catalog_name, verbose=False)
        clusters = self.cc.query_catalog(return_members=False)
        members = self.cc.query_catalog(return_members=True)

        idx = members.source_id.isin([gaia_params.source_id])
        if idx.sum() > 0:
            cluster_name = members.loc[idx, "Cluster"].squeeze()
            print(f"{self.target_name} is in {cluster_name}!")
        # check if vizier if known cluster/assoc member
        vizier_query = self.query_vizier_param("Assoc")
        assoc_from_literature = np.unique(list(vizier_query.values()))
        if len(assoc_from_literature) > 0:
            print("Cluster/assoc from literature:\n", assoc_from_literature)
            # if mem.Cluster.isin().sum():

        # estimate cluster parameter uncertainties from all members
        # err = pd.pivot_table(members, index=["Cluster"], aggfunc=np.median)[params]
        # err.columns = ['e_'+c for c in err_columns]
        # pd.merge(clusters, errs, on='index')
        g = members.groupby("Cluster")
        # add RV based on each cluster mean
        clusters = clusters.join(g.RV.mean(), on="Cluster")
        self.all_clusters = clusters

        # add error for each param
        param_errs = {}
        for param in params:
            name = "e_" + param
            if name not in members.columns:
                d = g[param].std()  # 1-sigma
                d.name = name
                param_errs[name] = d

                # join it as a new column in clusters
                clusters = clusters.join(d, on="Cluster")

        idxs = []
        for gparam, param in zip(gparams, params):
            star_mean = gaia_params[gparam]
            star_std = gaia_params[gparam + "_error"]

            # remove rows with large parameter uncertainty
            if frac_err is not None:
                idx1 = clusters.apply(
                    lambda x: (x["e_" + param] / x[param]) < frac_err, axis=1
                )
            #             print(f"{param}: {sum(~idx1)} removed")
            #             clusters[param].isnull().sum()/clusters.shape[0]
            else:
                idx1 = np.ones_like(clusters.index, dtype=bool)
            # replace those rows with nan to ignore
            d = clusters.where(idx1, np.nan)

            cluster_mean = d[param]
            cluster_std = d["e_" + param]

            idx2 = get_between_limits(
                lower=star_mean - star_std,
                upper=star_mean + star_std,
                data_mu=cluster_mean,
                data_sig=cluster_std,
                sigma=sigma,
            )
            if verbose:
                print(f"{param}: {idx2.sum()} matched.")
            idxs.append(idx2)

        # sum matches along row (cluster)
        nparams_match = np.sum(idxs, axis=0)
        # take the row with most match per parameter
        cluster_match_idx = nparams_match.argmax()
        cluster = clusters.iloc[cluster_match_idx]

        params_match_idx = np.array(idxs)[:, cluster_match_idx]
        params_match = np.array(params)[params_match_idx]
        if verbose:
            msg = f"matched {sum(params_match_idx)} params in {cluster.Cluster}:\n{params_match}"
            print(msg)

        if (pd.Series(params_match).isin(["ra", "dec"]).sum() == 2) & (
            len(params_match) > 3
        ):
            if return_idxs:
                return cluster, idxs
            else:
                return cluster
        else:
            print("Target not likely a cluster member.")

    def get_nearest_cluster_member(
        self,
        catalog_name="CantatGaudin2020",
        df=None,
        match_id=True,
        radius=None,
        with_parallax=True,
    ):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            cluster catalog to match against
        match_id : bool
            check if target gaiaid matches that of cluster member,
            else return nearest member only
        with_parallax : bool
            uses parallax to compute 3d distance; otherwise 2d
        radius : float
            search radius in arcsec (used when match_id=False)
        Returns
        -------
        match : pandas.Series
            matched cluster member by gaiaid
        """
        radius = self.search_radius if radius is None else radius * u.arcsec
        if (df is None) or (len(df) == 0):
            cc = ClusterCatalog(catalog_name=catalog_name)
            df = cc.query_catalog(return_members=True)
            # drop rows without specified Cluster name
            # df = df.dropna(subset=['Cluster'])

        if match_id:
            # return member with gaiaid identical to target
            if self.gaiaid is not None:
                idx = df.source_id.isin([self.gaiaid])
                if sum(idx) == 0:
                    errmsg = f"Gaia DR2 {self.gaiaid} not found in catalog\n"
                    errmsg += "Use match_id=False to get nearest cluster\n"
                    raise ValueError(errmsg)
                nearest_star = df.iloc[np.argmax(idx)]
                self.nearest_cluster_member = nearest_star
                if catalog_name != "Grandjean2020":
                    cluster_name = nearest_star["Cluster"]
                    assert (
                        str(cluster_name).lower() != "nan"
                    ), "Cluster name in catalog is nan"
                    self.nearest_cluster_name = cluster_name
                    self.nearest_cluster_members = df.loc[
                        df.Cluster == cluster_name
                    ]

                    if self.verbose and np.any(idx):
                        print(
                            f"Target is in {self.nearest_cluster_name} ({catalog_name})!"
                        )
                # return a series
                return df.iloc[np.argmax(idx)]
            else:
                errmsg = "Supply id via Target(gaiaDR2id=id) "
                errmsg += (
                    "or `query_gaia_dr2_catalog(return_nearest_xmatch=True)`"
                )
                raise ValueError(errmsg)
        else:
            # return closest member
            if "parallax" in df.columns:
                # Bouma and CantatGaudin2018 have parallaxes in members table
                # retain non-negative parallaxes including nan
                df = df[(df["parallax"] >= 0) | (df["parallax"].isnull())]
                cluster_mem_coords = SkyCoord(
                    ra=df["ra"].values * u.deg,
                    dec=df["dec"].values * u.deg,
                    distance=Distance(parallax=df["parallax"].values * u.mas),
                )
                if with_parallax:
                    if self.target_coord.distance is None:
                        # query distance
                        if self.verbose:
                            print(
                                "Querying parallax from Gaia DR2 to get distance..."
                            )
                        self.target_coord = get_target_coord_3d(
                            self.target_coord
                        )
                    # compute 3d distance between target and all cluster members
                    separations = cluster_mem_coords.separation_3d(
                        self.target_coord
                    )
                else:
                    cluster_mem_coords = SkyCoord(
                        ra=df["ra"].values * u.deg,
                        dec=df["dec"].values * u.deg,
                    )
                    # compute 2d distance between target and all cluster members
                    separations = cluster_mem_coords.separation(
                        self.target_coord
                    )
            else:
                # Babusiaux2018 does not have parallax in members table
                cluster_mem_coords = SkyCoord(
                    ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg
                )
                # compute 2d distance between target and all cluster members
                separations = cluster_mem_coords.separation(self.target_coord)

            nearest_star = df.iloc[separations.argmin()]
            self.distance_to_nearest_cluster_member = separations.min()
            self.nearest_cluster_member = nearest_star
            if radius < self.distance_to_nearest_cluster_member:
                print(
                    f"separation ({separations.min().arcsec:.1f} arcsec) > {radius}"
                )
            if catalog_name != "Grandjean2020":
                cluster_name = nearest_star.Cluster
                self.nearest_cluster_name = cluster_name
                if df is None:
                    df = Cluster(
                        cluster_name, mission=self.mission, verbose=False
                    ).query_cluster_members()
                # make sure only one cluster
                idx = df.Cluster == cluster_name
                self.nearest_cluster_members = df.loc[idx]
        return nearest_star

    def query_mast(self, radius=3):
        """
        https://astroquery.readthedocs.io/en/latest/mast/mast.html

        See also:
        https://gist.github.com/arfon/5cfc25d91ca21b8de62d64af7a0d25da
        and
        https://archive.stsci.edu/vo/python_examples.html
        """
        radius = radius * u.arcsec if radius is not None else 3 * u.arcsec
        if self.verbose:
            print(
                f"Searching MAST for ({self.target_coord.to_string()}) with radius={radius}."
            )
        table = Observations.query_region(self.target_coord, radius=radius)
        if table is None:
            print("No result from MAST.")
        else:
            df = table.to_pandas()
            if self.verbose:
                wavs = df.wavelength_region.dropna().unique()
                data = (
                    (df["obs_collection"] + "/" + df["filters"])
                    .dropna()
                    .unique()
                )
                print(f"Available data: {list(data)} in {list(wavs)}")
            return df

    def query_simbad(self, radius=3):
        """
        Useful to get literature values for spectral type, Vsini, etc.
        See:
        https://astroquery.readthedocs.io/en/latest/simbad/simbad.html
        See also meaning of object types (otype) here:
        http://simbad.u-strasbg.fr/simbad/sim-display?data=otypes
        """
        radius = radius * u.arcsec if radius is not None else 3 * u.arcsec
        if self.verbose:
            print(
                f"Searching MAST for ({self.target_coord}) with radius={radius}."
            )
        simbad = Simbad()
        simbad.add_votable_fields("typed_id", "otype", "sptype", "rot", "mk")
        table = simbad.query_region(self.target_coord, radius=radius)
        if table is None:
            print("No result from Simbad.")
        else:
            df = table.to_pandas()
            df = df.drop(
                [
                    "RA_PREC",
                    "DEC_PREC",
                    "COO_ERR_MAJA",
                    "COO_ERR_MINA",
                    "COO_ERR_ANGLE",
                    "COO_QUAL",
                    "COO_WAVELENGTH",
                ],
                axis=1,
            )
            return df

    def query_vizier(self, radius=3, verbose=None):
        """
        Useful to get relevant catalogs from literature
        See:
        https://astroquery.readthedocs.io/en/latest/vizier/vizier.html
        """
        verbose = self.verbose if verbose is None else verbose
        radius = self.search_radius if radius is None else radius * u.arcsec
        if verbose:
            print(
                f"Searching Vizier: ({self.target_coord.to_string()}) with radius={radius}."
            )
        # standard column sorted in increasing distance
        v = Vizier(
            columns=["*", "+_r"],
            # column_filters={"Vmag":">10"},
            # keywords=['stars:white_dwarf']
        )
        tables = v.query_region(self.target_coord, radius=radius)
        if tables is None:
            print("No result from Vizier.")
        else:
            if verbose:
                print(f"{len(tables)} tables found.")
                pprint(
                    {k: tables[k]._meta["description"] for k in tables.keys()}
                )
            self.vizier_tables = tables
            return tables

    def query_vizier_param(self, param=None, radius=3):
        """looks for value of param in each vizier table
        """
        if self.vizier_tables is None:
            tabs = self.query_vizier(radius=radius, verbose=False)
        else:
            tabs = self.vizier_tables

        if param is not None:
            idx = [param in i.columns for i in tabs]
            vals = {
                tabs.keys()[int(i)]: tabs[int(i)][param][0]
                for i in np.argwhere(idx).flatten()
            }
            if self.verbose:
                print(f"Found {sum(idx)} references in Vizier with [{param}].")
            return vals
        else:
            cols = [i.to_pandas().columns.tolist() for i in tabs]
            print(f"Choose parameter:\n{list(np.unique(flatten_list(cols)))}")

    def query_vizier_mags(
        self,
        catalogs=["apass9", "gaiadr2", "2mass", "wise", "epic"],
        add_err=True,
    ):
        """
        TODO: use sedfitter
        """
        if self.vizier_tables is None:
            tabs = self.query_vizier(verbose=False)
        else:
            tabs = self.vizier_tables
        refs = {
            # "tycho": {"tabid": "I/259/tyc2", "cols": ["BTmag", "VTmag"]},
            "apass9": {"tabid": "II/336/apass9", "cols": ["Bmag", "Vmag"]},
            "gaiadr2": {
                "tabid": "I/345/gaia2",
                "cols": ["Gmag", "BPmag", "RPmag"],
            },
            "2mass": {"tabid": "II/246/out", "cols": ["Jmag", "Hmag", "Kmag"]},
            "wise": {
                "tabid": "II/328/allwise",
                "cols": ["W1mag", "W2mag", "W3mag", "W4mag"],
                "epic": {"tabid": "IV/34/epic", "cols": "Kpmag"},
            },
        }

        phot = []
        for cat in catalogs:
            if cat in refs.keys():
                tabid = refs[cat]["tabid"]
                cols = refs[cat]["cols"]
                if tabid in tabs.keys():
                    d = tabs[tabid].to_pandas()[cols]
                    phot.append(d)
                    if add_err:
                        ecols = ["e_" + col for col in refs[cat]["cols"]]
                        if cat != "tycho":
                            e = tabs[tabid].to_pandas()[ecols]
                            phot.append(e)
                else:
                    print(f"No {cat} data in vizier.")
        d = pd.concat(phot, axis=1).squeeze()
        d.name = self.target_name
        return d

    def query_eso(self, diameter=3, instru=None, min_snr=1):
        """
        search spectra in ESO database
        """
        try:
            import pyvo as vo

            ssap_endpoint = "http://archive.eso.org/ssap"
            ssap_service = vo.dal.SSAService(ssap_endpoint)
        except Exception:
            raise ModuleNotFoundError("pip install pyvo")

        diameter = diameter * u.arcsec if diameter else 5 * u.arcsec

        if self.verbose:
            print(
                f"Searching ESO: ({self.target_coord.to_string()}) with diameter={diameter}."
            )
        ssap_resultset = ssap_service.search(
            pos=self.target_coord, diameter=diameter
        )

        table = ssap_resultset.to_table()
        if len(table) > 0:
            df = table.to_pandas()

            # decode bytes to str
            df["COLLECTION"] = df["COLLECTION"].apply(lambda x: x.decode())
            df["dp_id"] = df["dp_id"].apply(lambda x: x.decode())
            df["CREATORDID"] = df["CREATORDID"].apply(lambda x: x.decode())
            df["access_url"] = df["access_url"].apply(lambda x: x.decode())
            df["TARGETNAME"] = df["TARGETNAME"].apply(lambda x: x.decode())

            print(
                "Available data:\n{: <10} {: <10}".format(
                    "Instrument", "Nspectra"
                )
            )
            for k, d in df.groupby("COLLECTION"):
                print("{: <10} {: <10}".format(k, len(d)))

            fields = [
                "COLLECTION",
                "TARGETNAME",
                "s_ra",
                "s_dec",
                "APERTURE",
                "em_min",
                "em_max",
                "SPECRP",
                "SNR",
                "t_min",
                "t_max",
                "CREATORDID",
                "access_url",
                "dp_id",
            ]

            # appply filters
            if instru is not None:
                idx1 = (df["COLLECTION"] == instru).values
            else:
                idx1 = True
                instru = df["COLLECTION"].unique()
            filter = idx1 & (df["SNR"] > min_snr).values
            df = df.loc[filter, fields]
            if len(df) == 0:
                raise ValueError("No ESO data found.\n")
            elif len(df) > 0:
                # if verbose:
                print(
                    f"\nFound {len(df)} {instru} spectra with SNR>{min_snr}.\n"
                )
                targetnames = (
                    df["TARGETNAME"]
                    .apply(lambda x: str(x).replace("-", ""))
                    .unique()
                )
                if len(targetnames) > 1:
                    print("There are {} matches:".format(len(targetnames)))
                    # print coordinates of each match to check
                    for name in targetnames:
                        try:
                            coord = SkyCoord.from_name(name)
                            print(f"{name: <10}: ra,dec=({coord.to_string()})")
                        except Exception:
                            print(f"{name: <10}: failed to fetch coordinates")
                # if self.verbose:
                #     print('\nPreview:\n')
                #     print(df[["TARGETNAME", "s_ra", "s_dec", "APERTURE", \
                #           "em_min", "em_max", "SPECRP", "SNR", "t_min", "t_max"]].head())
                return df
            else:
                print("No data matches the given criteria.")
        else:
            print("No result from ESO.")

    def query_harps_bank_table(self, **kwargs):
        """
        search HARPS RV from Trifonov's database
        """
        if self.harps_bank_table is None:
            df = get_harps_bank(self.target_coord, **kwargs)
        else:
            df = self.harps_bank_table.copy()
        self.harps_bank_table = df
        return df

    def query_harps_rv(self, save_csv=True):
        """
        For column meanings:
        https://www2.mpia-hd.mpg.de/homes/trifonov/HARPS_RVBank_header.txt

        DRS : Data Reduction Software (pipeline)
        SERVAL : SpEctrum Radial Velocity AnaLyser (new pipeline)

        NZP : nightly zero point

        Activity indicators
        -------------------
        Halpha : H-alpha index
        NaD1 : Na DI index
        NaD2 : Na DII index
        dLW : differential line width to
        measure variations in the spectral line widths;
        CRX: chromatic RV index of the spectra
        to measure wavelength dependence of the RV from
        individual spectral orders as induced by e.g. spots;
        MLCRX : ML Chromatic index (Slope over log wavelength)

        RV contributions
        ----------------
        SNR_DRS : Signal-to-noise ratio in order 55
        BERV : Barycentric Earth radial velocity
        DRIFT : drift measure
        dNZP_mlc : Contribution from intra-night systematics
        SA : Contribution from secular acceleration
        CONTRAST_DRS
        BIS : Bisector span
        f_RV : observation flag

        Note the 2015 instrumental RV jump (fiber upgrade);
        intra-night drifts in DRS RVs <2015;
        https://arxiv.org/pdf/2001.05942.pdf
        """
        data_url = "https://www2.mpia-hd.mpg.de/homes/trifonov"
        table = self.query_harps_bank_table()
        targets = table["Target"].values
        filename = f"{targets[0]}_harps_all-data_v1.csv"
        local_fp = Path(DATA_PATH, filename)
        if local_fp.exists():
            fp = local_fp
            delimiter = ","
        else:
            fp = f"{data_url}/{targets[0]}_RVs/{filename}"
            delimiter = ";"
        try:
            df = pd.read_csv(fp, delimiter=delimiter)
            if not local_fp.exists():
                if save_csv:
                    df.to_csv(local_fp, index=False)
                    print("Saved: ", local_fp)
            else:
                print("Loaded: ", local_fp)
        except Exception as e:
            print(e)
            print(f"Check url: {fp}")
        self.harps_bank_rv = df
        self.validate_harps_rv()
        self.harps_bank_target_name = self.harps_bank_table.Target.unique()[0]
        return df

    def validate_harps_rv(self):
        if self.harps_bank_rv is None:
            raise ValueError("Try self.query_harps_rv()")
        else:
            rv = self.harps_bank_rv.copy()
        assert ((rv.BJD - rv.BJD_DRS) < 0.1).any()
        daytime = rv["f_RV"] == 32
        if sum(daytime) > 0:
            print(
                f"{sum(daytime)} out of {len(daytime)} are not within nautical twilight."
            )
        low_snr = rv["f_RV"] == 64
        if sum(low_snr) > 0:
            print(f"{sum(low_snr)} out of {len(low_snr)} have low SNR.")
        too_hi_snr = rv["f_RV"] == 124
        if sum(too_hi_snr) > 0:
            print(
                f"{sum(too_hi_snr)} out of {len(too_hi_snr)} have too high SNR."
            )
        if self.verbose:
            print("harps bank data validated.")

    def plot_harps_rv_scatter(self, data_type="rv"):
        """ """
        if self.harps_bank_rv is None:
            rv = self.query_harps_rv()
            assert rv is not None
        else:
            rv = self.harps_bank_rv.copy()

        if data_type == "rv":
            ncols, nrows, figsize = 3, 2, (9, 6)
            columns = [
                "RV_mlc_nzp",
                "RV_drs_nzp",
                "RV_mlc",
                "RV_drs",
                "RV_mlc_j",
                "SNR_DRS",
            ]
            title = f" HARPS RV bank: {self.harps_bank_target_name}"
        elif data_type == "activity":
            ncols, nrows, figsize = 3, 3, (9, 9)
            columns = [
                "CRX",
                "dLW",
                "Halpha",
                "NaD1",
                "NaD2",
                "FWHM_DRS",
                "CONTRAST_DRS",
                "BIS",
                "MLCRX",
            ]
            title = f"{self.harps_bank_target_name} Activity indicators"
        else:
            raise ValueError("Use rv or activity")
        fig, ax = pl.subplots(
            nrows,
            ncols,
            figsize=figsize,
            constrained_layout=True,
            # sharex=True
        )
        ax = ax.flatten()

        n = 0
        bjd0 = rv.BJD.astype(int).min()
        for col in columns:
            e_col = "e_" + col
            if e_col in rv.columns:
                ax[n].errorbar(
                    rv.BJD - bjd0,
                    rv[col],
                    yerr=rv[e_col],
                    marker=".",
                    label=col,
                    ls="",
                )
            else:
                ax[n].plot(
                    rv.BJD - bjd0, rv[col], marker=".", label=col, ls=""
                )
            ax[n].set_xlabel(f"BJD-{bjd0}")
            ax[n].set_ylabel(col)
            n += 1
        fig.suptitle(title)
        return fig

    def plot_harps_pairplot(self, columns=None):
        try:
            import seaborn as sb
        except Exception:
            raise ModuleNotFoundError("pip install seaborn")
        if self.harps_bank_rv is None:
            rv = self.query_harps_rv()
            assert rv is not None
        else:
            rv = self.harps_bank_rv.copy()

        if columns is None:
            columns = [
                "RV_mlc_nzp",
                "RV_drs_nzp",
                "CRX",
                "dLW",
                "Halpha",
                "NaD1",
                "NaD2",
                "FWHM_DRS",
                "BIS",
            ]
        # else:
        #     cols = rv.columns
        #     idx = cols.isin(columns)
        #     errmsg = f"{cols[idx]} column not in\n{cols.tolist()}"
        #     assert np.all(idx), errmsg
        g = sb.PairGrid(rv[columns], diag_sharey=False)
        g.map_upper(sb.scatterplot)
        g.map_lower(sb.kdeplot, colors="C0")
        g.map_diag(sb.kdeplot, lw=2)
        return g

    def plot_harps_rv_gls(
        self,
        columns=None,
        Porb=None,
        Prot=None,
        plims=(0.5, 27),
        use_period=True,
    ):
        """
        plims : tuple
            period limits (min,max)

        See Fig. 16 in https://arxiv.org/pdf/2001.05942.pdf
        """
        if self.harps_bank_rv is None:
            rv = self.query_harps_rv()
            assert rv is not None
        else:
            rv = self.harps_bank_rv.copy()

        if columns is None:
            columns = [
                "RV_mlc_nzp",
                "RV_drs_nzp",
                "CRX",
                "dLW",
                "Halpha",
                "NaD1",
                "NaD2",
                "FWHM_DRS",
                "CONTRAST_DRS",
                "BIS",
            ]
        fig, axs = pl.subplots(len(columns), 1, figsize=(10, 10), sharex=True)
        ax = axs.flatten()
        if self.verbose:
            print(
                f"Computing generalized Lomb-Scargle periodograms:\n{columns}"
            )
        for n, col in enumerate(tqdm(columns)):
            err = rv["e_" + col] if "e_" + col in rv.columns else None
            data = (rv.BJD, rv[col], err)
            gls = Gls(data, Pbeg=plims[0], Pend=plims[1], ofac=2)

            fbest = gls.best["f"]
            # T0 = gls.best["T0"]

            ax[n].plot(
                1 / gls.f if use_period else gls.f,
                gls.power,
                "b-",
                linewidth=0.5,
                c=f"C{n+1}",
                label=col,
            )
            # mark the highest peak
            ax[n].plot(
                1 / fbest if use_period else fbest,
                gls.power[gls.p.argmax()],
                "r.",
                label=f"$P = {1/fbest:.2f}$",
            )

            Porb = self.toi_period if Porb is None else Porb
            if Porb is not None:
                ax[n].axvline(
                    Porb if use_period else 1 / Porb, 0, 1, c="k", ls="-", lw=2
                )
            if Prot is not None:
                ax[n].axvline(
                    Prot if use_period else 1 / 1 / Prot,
                    0,
                    1,
                    c="k",
                    ls="--",
                    lw=2,
                )
            if plims is not None:
                assert isinstance(plims, tuple)
            ax[n].set_xlim(*plims)
            ax[n].legend(loc=0)
        if Porb is not None:
            ax[0].annotate("Porb", xy=(Porb, ax[0].get_ylim()[1]))
        if Prot is not None:
            ax[0].annotate("Prot", xy=(Prot, ax[0].get_ylim()[1]))
        ax[len(columns) // 2].set_ylabel("Lomb-Scargle Power")
        ax[n].set_xlabel("Period (days)")
        fig.subplots_adjust(hspace=0)
        return fig

    def plot_harps_rv_corr_matrix(self, columns=None):
        try:
            import seaborn as sb
        except Exception:
            raise ModuleNotFoundError("pip install seaborn")

        if self.harps_bank_rv is None:
            rv = self.query_harps_rv()
            assert rv is not None
        else:
            rv = self.harps_bank_rv.copy()

        if columns is None:
            columns = [
                "RV_mlc_nzp",
                "RV_drs_nzp",
                "CRX",
                "dLW",
                "Halpha",
                "NaD1",
                "NaD2",
                "FWHM_DRS",
                "CONTRAST_DRS",
                "BIS",
            ]
        # compute correlation
        corr = rv[columns].corr()

        # generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        fig, ax = pl.subplots(1, 1, figsize=(10, 10))

        # draw the heatmap with the mask and correct aspect ratio
        ax = sb.heatmap(
            corr,
            mask=mask,
            vmax=0.3,  # cmap=cmap,
            square=True,
            xticklabels=corr.index,
            yticklabels=corr.columns,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5, "label": "correlation"},
            ax=ax,
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)
        return fig

    def get_k2_data_from_exofop(self, table="star"):
        """
        """
        return get_k2_data_from_exofop(self.epicid, table=table)

    def query_specs_from_tfop(self, clobber=None, mission=None):
        """
        """
        mission = self.mission if mission is None else mission.lower()
        base = f"https://exofop.ipac.caltech.edu/{mission}"
        if mission == "tess":
            clobber = clobber if clobber is not None else self.clobber
            specs_table = get_specs_table_from_tfop(
                clobber=clobber, verbose=self.verbose
            )
            if self.ticid is None:
                print(
                    f"TIC ID of {self.target_name} will be inferred by cross-matching coordinates in TIC catalog."
                )
                ticid = self.query_tic_catalog(return_nearest_xmatch=True)
            else:
                ticid = self.ticid

            idx = specs_table["TIC ID"].isin([ticid])
            if self.verbose:
                url = base + f"/target.php?id={ticid}"
                print(f"There are {idx.sum()} spectra in {url}\n")
            return specs_table[idx]
        else:
            url = base + f"/edit_target.php?id={self.epicid}"
            print(f"Scraping not yet implemented.\nSee {url}")

    @property
    def toi_Tmag(self):
        return None if self.toi_params is None else self.toi_params["TESS Mag"]

    @property
    def toi_Tmag_err(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["TESS Mag err"]
        )

    @property
    def toi_period(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Period (days)"]
        )

    @property
    def toi_epoch(self):
        return (
            None if self.toi_params is None else self.toi_params["Epoch (BJD)"]
        )

    @property
    def toi_duration(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Duration (hours)"]
        )

    @property
    def toi_period_err(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Period (days) err"]
        )

    @property
    def toi_epoch_err(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Epoch (BJD) err"]
        )

    @property
    def toi_duration_err(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Duration (hours) err"]
        )

    @property
    def toi_depth(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Depth (ppm)"] * 1e-6
        )

    @property
    def toi_depth_err(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Depth (ppm) err"] * 1e-6
        )

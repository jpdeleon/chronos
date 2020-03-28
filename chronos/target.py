# -*- coding: utf-8 -*-

r"""
classes for searching object
"""

# Import standard library
from os.path import join, exists
import warnings

# from typing import Tuple
import logging
import re

# Import modules
# from matplotlib.figure import Figure
# from matplotlib.image import AxesImage
# from loguru import logger
import numpy as np
import pandas as pd
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs, Observations, tesscut
from astropy.coordinates import SkyCoord, Distance
from astropy import units as u
import lightkurve as lk

# from lightkurve.search import _query_mast as query_mast
# Import from package
from chronos import cluster
from chronos.utils import (
    get_all_campaigns,
    get_all_sectors,
    get_tess_ccd_info,
    get_toi,
    get_tois,
    get_target_coord,
    get_target_coord_3d,
    get_harps_RV,
    get_specs_table_from_tfop,
)

log = logging.getLogger(__name__)

__all__ = ["Target"]


class Target:
    def __init__(
        self,
        name=None,
        toiid=None,
        ticid=None,
        epicid=None,
        gaiaDR2id=None,
        ra_deg=None,
        dec_deg=None,
        search_radius=3,
        verbose=True,
        clobber=False,
        mission="tess",
    ):
        """
        handles target resolution and basic catalog queries

        Attributes
        ----------
        search_radius : float
            search radius for matching [arcsec]
        """
        self.toiid = toiid  # e.g. 837
        self.ticid = ticid  # e.g. 364107753
        self.epicid = epicid  # 201270176
        self.gaiaid = gaiaDR2id  # e.g. Gaia DR2 5251470948229949568
        self.target_name = name  # e.g. Pi Mensae
        self.search_radius = search_radius * u.arcsec
        self.tic_params = None
        self.gaia_params = None
        self.gaia_sources = None
        self.toi_params = None
        self.gmag = None
        # gaiaid match
        # self.cluster_member = None
        # self.cluster_members = None
        # self.cluster_name = None
        # position match
        self.distance_to_nearest_cluster_member = None
        self.nearest_cluster_member = None
        self.nearest_cluster_members = None
        self.nearest_cluster_name = None
        self.clobber = clobber
        self.verbose = verbose
        self.mission = mission.lower()
        self.vizier_tables = None

        if name:
            if name[:4].lower() == "epic":
                if epicid is None:
                    self.epicid = int(name.strip()[4:])
            elif name[:3].lower() == "tic":
                if ticid is None:
                    self.ticid = int(name.strip()[3:])
            elif name[:4].lower() == "gaia":
                if gaiaDR2id is None:
                    self.gaiaid = int(name.strip()[7:])

        # self.mission = mission
        self.ra = ra_deg
        self.dec = dec_deg
        # self.distance = None
        if (self.ticid is not None) and (self.toiid is None):
            tois = get_tois(clobber=True, verbose=False)
            idx = tois["TIC ID"].isin([self.ticid])
            if sum(idx) > 0:
                self.toiid = tois.loc[idx, "TOI"].values[0]
                if self.verbose:
                    print(f"TIC {self.ticid} is TOI {int(self.toiid)}!")
        if self.toiid is not None:
            self.toi_params = get_toi(
                toi=self.toiid, clobber=self.clobber, verbose=False
            ).iloc[0]
        if (self.ticid is None) and (self.toiid is not None):
            self.ticid = int(self.toi_params["TIC ID"])
        self.target_coord = get_target_coord(
            ra=self.ra,
            dec=self.dec,
            toi=self.toiid,
            tic=self.ticid,
            epic=self.epicid,
            gaiaid=self.gaiaid,
            name=self.target_name,
        )
        if self.mission == "tess":
            self.all_sectors = get_all_sectors(self.target_coord)
            self.tess_ccd_info = get_tess_ccd_info(self.target_coord)
        elif self.mission == "k2":
            # raise NotImplementedError
            self.all_campaigns = get_all_campaigns(self.epicid)

    def query_gaia_dr2_catalog(
        self, radius=None, return_nearest_xmatch=False, verbose=None
    ):
        """
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

        From Carillo+2019:
        2. the sample with the low parallax errors i.e. 0 < f < 0.1,
        has distances derived from simply inverting the parallax

        Whereas, the sample with higher parallax errors i.e. f > 0.1
        has distances derived from a Bayesian analysis following Bailer-Jones (2015),
        where they use a weak distance prior (i.e. exponentially decreasing space
        density prior) that changes with Galactic latitude and longitude

        3. See also Gaia DR2 Cross-match for the celestial reference system (ICRS)
        https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_proc/ssec_cu3ast_proc_xmatch.html
        and
        https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_cali/ssec_cu3ast_cali_frame.html
        """
        radius = self.search_radius if radius is None else radius * u.arcsec
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            # silenced when verbose=False instead of None
            print(
                f"""Querying Gaia DR2 catalog for ra,dec=({self.target_coord.to_string()}) within {radius}.\n"""
            )
        # load gaia params for all TOIs
        tab = Catalogs.query_region(
            self.target_coord, radius=radius, catalog="Gaia", version=2
        ).to_pandas()
        tab["source_id"] = tab.source_id.astype(int)

        # check if results from DR2 (epoch 2015.5)
        assert np.all(
            tab["ref_epoch"].isin([2015.5])
        ), "Epoch not 2015 (version<2?)"
        assert len(tab) > 0, f"use radius>{radius}"

        if np.any(tab["parallax"] < 0):
            # use positive parallaxes only
            # tab = tab[tab["parallax"]> 0] #this drops NaN too
            tab = tab[(tab["parallax"] >= 0) | (tab["parallax"].isnull())]
        """
        check parallax error here and apply corresponding distance calculation: see Note 1
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
        target_dist = Distance(parallax=star["parallax"] * u.mas)
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
            if self.gaiaid is not None:
                id = int(tab.iloc[0]["source_id"])
                msg = f"Nearest match ({id}) != {self.gaiaid}"
                assert int(self.gaiaid) == id, msg
            else:
                self.gaiaid = int(tab.iloc[0]["source_id"])
            self.gaia_params = tab.iloc[0]
            self.gmag = tab.iloc[0]["phot_g_mean_mag"]
            return tab.iloc[0]  # return series of len 1
        else:
            # if self.verbose:
            #     d = self.get_nearby_gaia_sources()
            #     print(d)
            self.gaia_sources = tab
            return tab  # return dataframe of len 2 or more

    def query_tic_catalog(self, radius=None, return_nearest_xmatch=False):
        """
        Query TIC v8 catalog from MAST

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
                f"Querying TIC catalog for ra,dec=({self.target_coord.to_string()}) within {radius}.\n"
            )
        # NOTE: check tic version
        tab = Catalogs.query_region(
            self.target_coord, radius=radius, catalog="TIC"
        ).to_pandas()

        nsources = len(tab)
        if return_nearest_xmatch or (nsources == 1):
            if nsources > 1:
                print(f"There are {nsources} TIC stars within {radius}")
            # get nearest match
            tab = tab.iloc[0]
            if tab.wdflag == 1:
                print(f"white dwarf flag = True!")
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
        if self.gaiaid is not None:
            assert g.source_id == int(t["GAIA"]), "Different source IDs"

        msg = "Gaia and TIC catalog cross-match succeeded."
        print(msg)
        return True

    def get_nearby_gaia_sources(self, radius=60, add_column=None):
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

        if self.gaia_sources is None:
            d = self.query_gaia_dr2_catalog(radius=radius)
        else:
            d = self.gaia_sources.copy()

        if self.gaiaid is None:
            # nearest match (first entry row=0) is assumed as the target
            gaiaid = int(d.iloc[0]["source_id"])
        else:
            gaiaid = self.gaiaid
        msg = f"Only 1 gaia source found within r={radius} arcsec"
        assert isinstance(d, pd.DataFrame), msg

        idx = d.source_id == gaiaid
        target_gmag = d.loc[idx, "phot_g_mean_mag"]
        d["distance"] = d["distance"].apply(
            lambda x: x * u.arcmin.to(u.arcsec)
        )
        d["delta_Gmag"] = d["phot_g_mean_mag"] - target_gmag
        # compute dilution factor
        d["gamma"] = 1 + 10 ** (0.4 * d["delta_Gmag"])
        columns = [
            "source_id",
            "distance",
            "parallax",
            "phot_g_mean_mag",
            "delta_Gmag",
            "gamma",
        ]
        if add_column is not None:
            assert (isinstance(add_column, str)) & (add_column in d.columns)
            columns.append(add_column)
        return d[columns]

    def compute_Tmax_from_depth(self, depth=None):
        """
        """
        if self.toi_params is None:
            toi_params = self.get_toi(clobber=False, verbose=False).iloc[0]
        else:
            toi_params = self.toi_params

        depth = toi_params["Depth (ppm)"] * 1e-6
        if self.tic_params is None:
            tic_params = self.query_tic_catalog(return_nearest_xmatch=True)
        else:
            tic_params = self.tic_params
        Tmag = tic_params["Tmag"]
        dT = 2.5 * np.log10(depth)
        Tmax = Tmag + dT
        if self.verbose:
            print(
                f"Given depth={depth*100:.4f}%, Tmag={Tmax:.2f} is the max. mag of a resolved companion that can reproduce this transit"
            )
        return Tmax

    def get_possible_NEBs(self, depth, gaiaid=None, kmax=1.0):
        """
        depth is useful to rule out deep eclipses when kmax/gamma>depth

        kmax : float [0,1]
            maximum eclipse depth (default=1)
        """
        assert (kmax >= 0.0) & (kmax <= 1.0), "eclipse depth is between 0 & 1"

        d = self.get_nearby_gaia_sources()

        good, bad = [], []
        for index, row in d.iterrows():
            id, dmag, gamma = row[["source_id", "delta_Gmag", "gamma"]]
            if int(id) != gaiaid:
                if depth > kmax / gamma:
                    # observed depth is too deep to have originated from the secondary star
                    good.append(id)
                else:
                    # uncertain signal source
                    bad.append(id)
        uncleared = d.loc[d.source_id.isin(bad)]
        return uncleared

    def get_nearest_cluster_member(
        self,
        catalog_name="Bouma2019",
        df=None,
        match_id=True,
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
        with_parallax : uses parallax to compute 3d distance; otherwise 2d

        Returns
        -------
        match : pandas.Series
            matched cluster member by gaiaid
        """
        if (df is None) or (len(df) == 0):
            cc = cluster.ClusterCatalog(catalog_name=catalog_name)
            df = cc.query_catalog(return_members=True)
            # drop rows without specified Cluster name
            # df = df.dropna(subset=['Cluster'])

        if match_id:
            # return member with gaiaid identical to target
            if self.gaiaid is not None:
                idx = df.source_id.isin([self.gaiaid])
                if sum(idx) == 0:
                    errmsg = f"Gaia DR2 {self.gaiaid} not found in catalog\n"
                    errmsg += f"Use match_id=False to get nearest cluster\n"
                    raise ValueError(errmsg)
                nearest_star = df.loc[idx]
                self.nearest_cluster_member = nearest_star
                cluster_name = nearest_star["Cluster"].values[0]
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
                return df.loc[idx]
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
                                f"Querying parallax from Gaia DR2 to get distance"
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
            cluster_name = nearest_star.Cluster
            self.nearest_cluster_name = cluster_name
            if df is None:
                df = cluster.Cluster(
                    cluster_name, verbose=False
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
                f"Searching MAST for ({self.target_coord.to_string()}) with radius={radius}"
            )
        table = Observations.query_region(self.target_coord, radius=radius)
        msg = "No results from MAST"
        assert len(table) > 0, msg
        df = table.to_pandas()
        if self.verbose:
            wavs = df.wavelength_region.dropna().unique()
            data = (
                (df["obs_collection"] + "/" + df["filters"]).dropna().unique()
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
                f"Searching MAST for ({self.target_coord}) with radius={radius}"
            )
        simbad = Simbad()
        simbad.add_votable_fields("typed_id", "otype", "sptype", "rot", "mk")
        table = simbad.query_region(self.target_coord, radius=radius)
        msg = "No result from Simbad"
        assert len(table) > 0, msg
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
        radius = 3 * u.arcsec if radius is None else radius * u.arcsec
        if self.verbose:
            print(
                f"Searching Vizier: ({self.target_coord.to_string()}) with radius={radius}"
            )
        # standard column sorted in increasing distance
        v = Vizier(
            columns=["*", "+_r"],
            # column_filters={"Vmag":">10"},
            # keywords=['stars:white_dwarf']
        )
        tables = v.query_region(self.target_coord, radius=radius)
        msg = "No result from Vizier"
        assert len(table) > 0, msg
        if verbose:
            print(f"{len(tables)} tables found.")
            print({k: v.description for k, v in tables.items()})
        self.vizier_tables = tables
        return tables

    def query_vizier_param(self, param, radius=3):
        if self.vizier_tables is None:
            tabs = self.query_vizier(radius=radius)
        else:
            tabs = self.vizier_tables
        idx = [param in i.columns for i in tabs]
        vals = {
            tabs.keys()[int(i)]: tabs[int(i)][param][0]
            for i in np.argwhere(idx).flatten()
        }
        if self.verbose:
            print(f"Found {sum(idx)} references with {param}")
        return vals

    def query_eso(self, diameter=3, instru=None, min_snr=1):
        """
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
                f"Searching ESO: ({self.target_coord.to_string()}) with diameter={diameter}"
            )
        ssap_resultset = ssap_service.search(
            pos=self.target_coord, diameter=diameter
        )

        table = ssap_resultset.to_table()
        msg = "No results from ESO"
        assert len(table) > 0, msg
        df = table.to_pandas()

        # decode bytes to str
        df["COLLECTION"] = df["COLLECTION"].apply(lambda x: x.decode())
        df["dp_id"] = df["dp_id"].apply(lambda x: x.decode())
        df["CREATORDID"] = df["CREATORDID"].apply(lambda x: x.decode())
        df["access_url"] = df["access_url"].apply(lambda x: x.decode())
        df["TARGETNAME"] = df["TARGETNAME"].apply(lambda x: x.decode())

        print(
            "Available data:\n{: <10} {: <10}".format("Instrument", "Nspectra")
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
            print(f"\nFound {len(df)} {instru} spectra with SNR>{min_snr}\n")
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
        else:
            raise ValueError("\nNo data that matches the given criteria.")

    def query_harps_rv(self, **kwargs):
        df = get_harps_RV(self.target_coord, **kwargs)
        return df

    def query_specs_from_tfop(self, clobber=None):
        """
        """
        base = "https://exofop.ipac.caltech.edu/tess/"
        clobber = clobber if clobber is not None else self.clobber
        specs_table = get_specs_table_from_tfop(
            clobber=clobber, verbose=self.verbose
        )
        if self.ticid is None:
            ticid = self.query_tic_catalog(return_nearest_xmatch=True)
        else:
            ticid = self.ticid

        idx = specs_table["TIC ID"].isin([ticid])
        if self.verbose:
            print(
                f"There are {idx.sum()} spectra in {base}target.php?id={ticid}\n"
            )
        return specs_table[idx]

    @property
    def toi_period(self):
        return self.toi_params["Period (days)"]

    @property
    def toi_epoch(self):
        return self.toi_params["Epoch (BJD)"]

    @property
    def toi_duration(self):
        return self.toi_params["Duration (hours)"]

    @property
    def toi_period_err(self):
        return self.toi_params["Period (days) err"]

    @property
    def toi_epoch_err(self):
        return self.toi_params["Epoch (BJD) err"]

    @property
    def toi_duration_err(self):
        return self.toi_params["Duration (hours) err"]

    @property
    def toi_depth(self):
        return self.toi_params["Depth (ppm)"] * 1e-6

# -*- coding: utf-8 -*-

r"""
classes for searching target and querying cluster catalogs

See also from astroquery.xmatch import XMatch
"""

# Import standard library
from os.path import join, exists
import logging
import re

# Import modules
import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator
from astroquery.mast import Catalogs
from astroquery.mast.tesscut import Tesscut
from astropy.coordinates import SkyCoord, Distance
from astropy import units as u
import lightkurve as lk
from tqdm import tqdm
import deepdish as dd

# from lightkurve.search import _query_mast as query_mast
# Import from package
from chronos.config import DATA_PATH
from chronos.utils import (
    get_toi,
    get_tois,
    get_target_coord,
    get_target_coord_3d,
    get_absolute_color_index,
    get_mamajek_table,
)

log = logging.getLogger(__name__)

__all__ = ["Target", "Cluster", "ClusterCatalog"]

CATALOG_LIST = [
    "Bouma2019",
    "Babusiaux2018",
    "CantatGaudin2018",
    "Bossini2019",
    "Dias2014",
    "Karchenko2013",
    "Cody2018",
]


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
        search_radius=3 * u.arcsec,
        verbose=True,
        clobber=False,
    ):
        self.toiid = toiid  # e.g. 837
        self.ticid = ticid  # e.g. 364107753
        self.epicid = epicid  # 201270176
        self.gaiaid = gaiaDR2id  # e.g. Gaia DR2 5251470948229949568
        self.target_name = name  # e.g. Pi Mensae
        self.search_radius = search_radius
        self.tic_params = None
        self.gaia_params = None
        self.toi_params = None
        self.gmag = None
        self.tesscut_tpf = None
        self.corrector = None
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
        # self.sector = sector
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
            )
        if (self.ticid is None) and (self.toiid is not None):
            self.ticid = int(self.toi_params["TIC ID"].values[0])
        self.target_coord = get_target_coord(
            ra=self.ra,
            dec=self.dec,
            toi=self.toiid,
            tic=self.ticid,
            epic=self.epicid,
            gaiaid=self.gaiaid,
            name=self.target_name,
        )
        self.ccd_info = Tesscut.get_sectors(self.target_coord).to_pandas()

    def get_all_sectors(self):
        """
        """
        df = self.ccd_info
        all_sectors = [int(i) for i in df["sector"].values]
        return all_sectors

    def get_sector_cam_ccd(self):
        """get TESS sector, camera, and ccd numbers using Tesscut
        """
        all_sectors = self.get_all_sectors()
        df = self.ccd_info
        if self.sector:
            sector = self.sector
            sector_idx = df["sector"][
                df["sector"].isin([self.sector])
            ].index.tolist()
            if len(sector_idx) == 0:
                raise ValueError(f"Available sector(s): {all_sectors}")
            cam = str(df.iloc[sector_idx]["camera"].values[0])
            ccd = str(df.iloc[sector_idx]["ccd"].values[0])
        else:
            sector_idx = 0
            sector = str(df.iloc[sector_idx]["sector"])
            cam = str(df.iloc[sector_idx]["camera"])
            ccd = str(df.iloc[sector_idx]["ccd"])
        return sector, cam, ccd

    def estimate_Av(self, map="sfd", constant=3.1):
        """
        compute the extinction Av from color index E(B-V)
        estimated from dustmaps via Av=constant*E(B-V)

        Parameters
        ----------
        map : str
            dust map; see https://dustmaps.readthedocs.io/en/latest/maps.html
        """
        try:
            import dustmaps
        except ImportError:
            print("pip install dustmaps")

        if map == "sfd":
            from dustmaps import sfd

            # sfd.fetch()
            dust_map = sfd.SFDQuery()
        elif map == "planck":
            from dustmaps import planck

            # planck.fetch()
            dust_map = planck.PlanckQuery()
        else:
            raise ValueError(f"Available maps: (sfd,planck)")

        ebv = dust_map(self.target_coord)
        Av = constant * ebv
        return Av

    def query_toi(self, toi=None, **kwargs):
        d = get_toi(toi=toi, **kwargs)
        if d is not None:
            self.toi_params = d
        return d

    def query_gaia_dr2_catalog(self, radius=None, return_nearest_xmatch=False):
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
        if self.verbose:
            print(
                f"""Querying Gaia DR2 catalog for {self.target_coord.to_string()} within {radius}.\n"""
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
        if self.gaiaid is not None:
            # find by id match for gaiaDR2id input
            idx = tab.source_id.isin([self.gaiaid]).argmax()
        else:
            # find by nearest distance (for toiid or ticid input)
            idx = self.target_coord.separation(gcoords).argmin()
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

        if return_nearest_xmatch or (len(tab) == 1):
            self.gaiaid = int(tab.loc[0, "source_id"])
            self.gaia_params = tab.iloc[0]
            self.gmag = self.gaia_params["phot_g_mean_mag"]
            return tab.iloc[0]  # return series of len 1
        else:
            # if self.verbose:
            #     d = self.get_nearby_gaia_sources()
            #     print(d)
            self.gaia_params = tab
            return tab  # return dataframe of len 2 or more

    def get_nearby_gaia_sources(self, radius=60.0, add_column=None):
        """
        get information about stars within 60 arcsec and
        dilution factor from delta Gmag

        Parameters
        ----------
        radius : float
            query radius in arcsec
        add_column : str
            additional Gaia column name to show (e.g. radial_velocity)
        """
        if self.gaia_params is None:
            _ = self.query_gaia_dr2_catalog(radius=60)
        if len(self.gaia_params) == 1:
            _ = self.query_gaia_dr2_catalog(radius=60)
        d = self.gaia_params.copy()

        if self.gaiaid is None:
            # nearest match (first entry row=0) is assumed as the target
            gaiaid = int(d.iloc[0]["source_id"])
        else:
            gaiaid = self.gaiaid
        idx = d.source_id == gaiaid
        target_gmag = d.loc[idx, "phot_g_mean_mag"].values[0]
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

    def get_possible_NEBs(self, depth, gaiaid=None, kmax=1.0):
        """
        depth is useful to rule out deep eclipses when kmax/gamma>depth

        kmax : float [0,1]
            maximum eclipse depth (default=1)
        """
        assert (kmax >= 0.0) & (kmax <= 1.0), "eclipse depth is between 0 & 1"

        d = self.get_nearby_gaia_sources()

        good, bad = [], []
        for id, dmag, gamma in d[["source_id", "delta_Gmag", "gamma"]].values:
            if int(id) != gaiaid:
                if depth > kmax / gamma:
                    # observed depth is too deep to have originated from the secondary star
                    good.append(id)
                else:
                    # uncertain signal source
                    bad.append(id)
        uncleared = d.loc[d.source_id.isin(bad)]
        return uncleared

    def query_tic_catalog(self, radius=None, return_nearest_xmatch=False):
        """
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
                f"""Querying TIC catalog for {self.target_coord.to_string()}
            within {radius}.\n"""
            )
        # NOTE: check tic version
        tab = Catalogs.query_region(
            self.target_coord, radius=radius, catalog="TIC"
        ).to_pandas()
        if return_nearest_xmatch:
            tab = tab.iloc[0]
        self.tic_params = tab
        return tab

    def find_nearest_cluster_member(
        self, catalog_name="Bouma2019", df=None, match_id=True
    ):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            cluster catalog to match against
        match_id : int
            check if target gaiaid matches that of cluster member,
            else return nearest member only

        Returns
        -------
        match : pandas.Series
            matched cluster member by gaiaid
        """
        if (df is None) or (len(df) == 0):
            cc = ClusterCatalog(catalog_name=catalog_name)
            df = cc.query_catalog(return_members=True)
            # drop rows without specified Cluster name
            # df = df.dropna(subset=['Cluster'])

        if match_id:
            # return member with gaiaid identical to target
            if self.gaiaid:
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
            cluster_mem_coords = SkyCoord(
                ra=df["ra"].values * u.deg,
                dec=df["dec"].values * u.deg,
                distance=df["distance"].values * u.pc,
            )
            if self.target_coord.distance is None:
                # query distance
                if self.verbose:
                    print(f"Querying parallax from Gaia DR2 to get distance")
                self.target_coord = get_target_coord_3d(self.target_coord)
            # compute 3d distance between target and all cluster members
            separations = cluster_mem_coords.separation_3d(self.target_coord)
            nearest_star = df.iloc[separations.argmin()]
            self.distance_to_nearest_cluster_member = separations.min()
            self.nearest_cluster_member = nearest_star
            cluster_name = nearest_star.Cluster
            self.nearest_cluster_name = cluster_name
            if df is None:
                df = Cluster(
                    cluster_name, verbose=False
                ).query_cluster_members()
            # make sure only one cluster
            idx = df.Cluster == cluster_name
            self.nearest_cluster_members = df.loc[idx]
        return nearest_star

    def get_spec_type(
        self,
        columns="Teff B-V J-H H-Ks".split(),
        nsamples=int(1e4),
        return_samples=False,
        clobber=False,
    ):
        """
        Interpolate spectral type from Mamajek table from
        http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
        based on observables Teff and color indices.

        Parameters
        ----------
        columns : list
            column names of input parameters
        nsamples : int
            number of Monte Carlo samples (default=1e4)
        clobber : bool (default=False)
            re-download Mamajek table

        Returns
        -------
        interpolated spectral type

        Notes:
        It may be good to check which color index yields most accurate result
        """
        df = get_mamajek_table(clobber=clobber, verbose=self.verbose)
        if self.gaia_params is None:
            self.gaia_params = self.query_gaia_dr2_catalog(
                return_nearest_xmatch=True
            )
        if self.tic_params is None:
            self.tic_params = self.query_tic_catalog(
                return_nearest_xmatch=True
            )

        # effective temperature
        col = "teff"
        teff = self.gaia_params[f"{col}_val"]
        siglo = (
            self.gaia_params[f"{col}_val"]
            - self.gaia_params[f"{col}_percentile_lower"]
        )
        sighi = (
            self.gaia_params[f"{col}_percentile_upper"]
            - self.gaia_params[f"{col}_val"]
        )
        uteff = np.sqrt(sighi ** 2 + siglo ** 2)
        s_teff = (
            teff + np.random.randn(nsamples) * uteff
        )  # Monte Carlo samples

        # B-V color index
        bv_color = self.tic_params["Bmag"] - self.tic_params["Vmag"]
        ubv_color = (
            self.tic_params["e_Bmag"] + self.tic_params["e_Vmag"]
        )  # uncertainties add
        s_bv_color = (
            bv_color + np.random.randn(nsamples) * ubv_color
        )  # Monte Carlo samples

        # J-H color index
        jh_color = self.tic_params["Jmag"] - self.tic_params["Hmag"]
        ujh_color = (
            self.tic_params["e_Jmag"] + self.tic_params["e_Hmag"]
        )  # uncertainties add
        s_jh_color = (
            jh_color + np.random.randn(nsamples) * ujh_color
        )  # Monte Carlo samples

        # H-K color index
        hk_color = self.tic_params["Hmag"] - self.tic_params["Kmag"]
        uhk_color = (
            self.tic_params["e_Hmag"] + self.tic_params["e_Kmag"]
        )  # uncertainties add
        s_hk_color = (
            hk_color + np.random.randn(nsamples) * uhk_color
        )  # Monte Carlo samples

        # Interpolate
        interp = NearestNDInterpolator(
            df[columns].values, df["#SpT"].values, rescale=False
        )
        samples = interp(s_teff, s_bv_color, s_jh_color, s_hk_color)
        # encode category
        spt_cats = pd.Series(samples, dtype="category")  # .cat.codes
        spt = spt_cats.mode().values[0]
        if return_samples:
            return spt, samples
        else:
            return spt

    def make_custom_ffi_lc(
        self,
        sector=None,
        cutout_size=(50, 50),
        mask_threshold=3,
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
        mask_threshold : float
            threshold (sigma) to create aperture mask
        pca_nterms : int
            number of pca terms to use

        Returns
        -------
        corrected_lc : lightkurve object
        """
        if sector is None:
            all_sectors = self.get_all_sectors()
            sector = all_sectors[0]
            print(f"Available sectors: {all_sectors}")
            print(f"sector {sector} is used.\n")
        if self.tesscut_tpf is not None:
            if self.tesscut_tpf.sector == sector:
                tpf = self.tesscut_tpf
            else:
                tpf = lk.search_tesscut(
                    self.target_coord, sector=sector
                ).download(cutout_size=cutout_size)
                self.tesscut_tpf = tpf
        else:
            tpf = lk.search_tesscut(self.target_coord, sector=sector).download(
                cutout_size=cutout_size
            )
            self.tesscut_tpf = tpf

        # remove zeros
        zero_mask = (tpf.flux_err == 0).all(axis=(1, 2))
        if zero_mask.sum() > 0:
            tpf = tpf[~zero_mask]
        # Make an aperture mask and a raw light curve
        aper = tpf.create_threshold_mask(threshold=mask_threshold)
        raw_lc = tpf.to_lightcurve(aperture_mask=aper)

        # Make a design matrix and pass it to a linear regression corrector
        regressors = tpf.flux[:, ~aper]
        dm = (
            lk.DesignMatrix(regressors, name="regressors")
            .pca(nterms=pca_nterms)
            .append_constant()
        )
        rc = lk.RegressionCorrector(raw_lc)
        # if remove_outliers:
        #     clean_lc, outlier_mask = tpf.to_lightcurve(
        #         aperture_mask=aper
        #     ).remove_outliers(return_mask=True)
        #     regressors = tpf.flux[:, ~aper][~outlier_mask]
        #     dm = lk.DesignMatrix(regressors, name="regressors").pca(nterms=pca_nterms).append_constant()
        #     rc = lk.RegressionCorrector(clean_lc)
        self.corrector = rc
        corrected_lc = rc.correct(dm)

        # Optional: Remove the scattered light, allowing for the large offset from scattered light
        if with_offset:
            corrected_lc = (
                raw_lc - rc.model_lc + np.percentile(rc.model_lc.flux, q=5)
            )
        return corrected_lc.normalize()


class ClusterCatalog:
    def __init__(
        self, catalog_name="Bouma2019", verbose=True, data_loc=DATA_PATH
    ):
        """Initialize the catalog

        Attributes
        ----------
        data_loc : str
            data directory
        all_catalogs: list
            list of all catalogs
        """
        self.catalog_name = catalog_name
        self.data_loc = data_loc
        self.verbose = verbose
        self.catalog_list = CATALOG_LIST
        self.all_members = None
        self.all_clusters = None

    def query_catalog(self, name=None, return_members=False):
        """Query catalogs

        Parameters
        ----------
        name : str
            catalog name; see `ClusterCatalog.all_catalogs`
        return_members : bool
            return parameters for all members instead of the default

        Returns
        -------
        df : pandas.DataFrame
            dataframe parameters of the cluster or its individual members
        """
        self.catalog_name = name if name is not None else self.catalog_name
        if self.catalog_name == "Bouma2019":
            if return_members:
                df_mem = self.get_members_Bouma2019()
                self.all_members = df_mem
                return df_mem
            else:
                df = self.get_clusters_Bouma2019()
                self.all_clusters = df
                return df
        elif self.catalog_name == "CantatGaudin2018":
            if return_members:
                df_mem = self.get_members_CantatGaudin2018()
                self.all_members = df_mem
                return df_mem
            else:
                df = self.get_clusters_CantatGaudin2018()
                self.all_clusters = df
                return df
        elif self.catalog_name == "Dias2014":
            if return_members:
                raise ValueError(
                    "No individual cluster members in Dias catalog"
                )
            else:
                df = self.get_clusters_Dias2002_2015()
                self.all_clusters = df
                return df
        elif self.catalog_name == "Bossini2019":
            if return_members:
                raise ValueError(
                    "No individual cluster members in Bossini age catalog"
                )
            else:
                df = self.get_clusters_Bossini2019()
                self.all_clusters = df
                return df
        elif self.catalog_name == "Babusiaux2018":
            if return_members:
                raise NotImplementedError("To be updated")
            else:
                df = self.get_clusters_Babusiaux2018()
                self.all_clusters = df
                return df
        elif self.catalog_name == "Karchenko2013":
            if return_members:
                raise ValueError(
                    "No individual cluster members in Karchenko age catalog"
                )
            else:
                df = self.get_clusters_Karchenko2013()
                self.all_clusters = df
                return df
        elif self.catalog_name in self.catalog_list:
            raise NotImplementedError("Catalog to be added later.")
        # add more catalogs here
        else:
            raise ValueError("Catalog name not found; see `all_catalogs`")

    def get_members_Bouma2019(self):
        """
        Bouma et al. 2019:
        https://ui.adsabs.harvard.edu/abs/2019arXiv191001133B/abstract
        """
        fp = join(
            self.data_loc, "TablesBouma2019/OC_MG_FINAL_v0.3_publishable.csv"
        )
        df = pd.read_csv(fp, header=0, sep=";")
        df = df.rename(
            columns={"cluster": "clusters", "unique_cluster_name": "Cluster"}
        )
        if np.any(df["parallax"] > 0):
            # df = df[df["parallax"] > 0] #this drops NaNs too
            df = df[(df["parallax"] >= 0) | (df["parallax"].isnull())]
            if self.verbose:
                print(f"Some parallaxes are negative in {self.catalog_name}!")
                print("These are removed for the meantime.")
                print("For proper treatment, see:")
                print("https://arxiv.org/pdf/1804.09366.pdf\n")

        icrs = SkyCoord(
            ra=df["ra"].values * u.deg,
            dec=df["dec"].values * u.deg,
            distance=Distance(parallax=df["parallax"].values * u.mas),
            #                 radial_velocity=df['RV'].values*u.km/u.s,
            pm_ra_cosdec=df["pmra"].values * u.mas / u.yr,
            pm_dec=df["pmdec"].values * u.mas / u.yr,
            frame="fk5",
            equinox="J2000.0",
        )
        # compute absolute G magnitude
        df["distance"] = icrs.distance
        # compute absolute Gmag
        # df["abs_gmag"] = get_absolute_gmag(df["phot_g_mean_mag"], df["distance"], df["a_g_val"])
        # compute intrinsic color index
        # df["bp_rp0"] = get_absolute_color_index(df["a_g_val"], df["phot_bp_mean_mag"], df["phot_rp_mean_mag"])
        # df['abs_gmag'].unit = u.mag
        return df

    def get_clusters_Bouma2019(self):
        """Bouma et al. 2019:
        https://ui.adsabs.harvard.edu/abs/2019arXiv191001133B/abstract

        Note: median values of cluster members are used but might be incorrect!
        """
        d = self.get_members_Bouma2019()
        # count unique cluster group members
        g = d.groupby("Cluster").groups
        members = pd.Series({k: len(g[k]) for k in g.keys()}, name="count")
        df = pd.pivot_table(d, index=["Cluster"], aggfunc=np.median)
        df = df.drop("source_id", axis=1)
        df = pd.merge(df, members, left_index=True, right_index=True)
        df = df.reset_index()
        return df

    def get_clusters_CantatGaudin2018(self):
        """Cantat-Gaudin et al. 2018:
        http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/618/A93
        """
        fp = join(
            self.data_loc,
            "TablesCantatGaudin2018/Table1_1229_open_clusters.tsv",
        )
        df = pd.read_csv(fp, delimiter="\t", comment="#")
        # remove spaces
        df.Cluster = df.Cluster.apply(lambda x: x.strip())
        df = df.rename(
            columns={
                "RAJ2000": "ra",
                "DEJ2000": "dec",
                "plx": "parallax",
                "pmRA": "pmra",
                "pmDE": "pmdec",
            }
        )
        return df

    def get_members_CantatGaudin2018(self):
        """Cantat-Gaudin et al. 2018:
        http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/618/A93
        """
        fp = join(
            self.data_loc,
            "TablesCantatGaudin2018/Membership probabilities of all individual stars.tsv",
        )
        df = pd.read_csv(fp, delimiter="\t", comment="#")
        df.Cluster = df.Cluster.apply(lambda x: x.strip())
        df = df.rename(
            columns={
                "RA_ICRS": "ra",
                "DE_ICRS": "dec",
                "Source": "source_id",
                "plx": "parallax",
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "Gmag": "phot_g_mean_mag",
            }
        )
        return df

    def get_clusters_Babusiaux2018(self):
        """Babusiaux, Gaia Collaboration et al. 2018, Table 2:
        http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/616/A10
        """
        fp = join(
            self.data_loc, "TablesGaiaDR2HRDpaper/Table2_32 open clusters.csv"
        )
        df = pd.read_csv(fp, delimiter=",", comment="#")
        df["distance"] = Distance(distmod=df["DM"]).pc
        df["Cluster"] = df.Cluster.apply(lambda x: x.replace(" ", "_"))
        df = df.rename(
            columns={
                "RA": "ra",
                "Dec": "dec",
                "log(age)": "log10_age",
                "DM": "dist_mod",
                # "Memb": ""
            }
        )
        return df

    def get_clusters_Babusiaux2018_near(self):
        """Babusiaux, Gaia Collaboration et al. 2018, Table 3 (<250 pc):
        http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/616/A10
        """
        fp = join(
            self.data_loc,
            "TablesGaiaDR2HRDpaper/Table3_Mean parameters for 9 open clusters within 250pc.tsv",
        )
        df = pd.read_csv(fp, delimiter="\t", comment="#")
        df.columns = [c.strip() for c in df.columns]
        df.Cluster = df.Cluster.apply(lambda x: x.strip())
        return df

    def get_clusters_Babusiaux2018_far(self):
        """Babusiaux, Gaia Collaboration et al. 2018, Table 4 (>250 pc):
        http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/616/A10
        """
        fp = join(
            self.data_loc,
            "TablesGaiaDR2HRDpaper/TableA4_Mean parameters for 37 open clusters beyond 250pc.tsv",
        )
        df = pd.read_csv(fp, delimiter="\t", comment="#")
        df = df.replace(r"^\s*$", np.nan, regex=True)
        df.columns = [c.strip() for c in df.columns]
        df.Cluster = df.Cluster.apply(lambda x: x.strip())
        return df

    def get_members_Babusiaux2018_near(self):
        """Babusiaux, Gaia Collaboration et al. 2018, Table A1a (<250 pc):
        http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/616/A10
        """
        fp = join(
            self.data_loc,
            "TablesGaiaDR2HRDpaper/TableA1a_9 open cluster members within 250 pc.csv",
        )
        df = pd.read_csv(fp, delimiter=",", comment="#")
        df.columns = [c.strip() for c in df.columns]
        df.Cluster = df.Cluster.apply(lambda x: x.strip())
        return df

    def get_members_Babusiaux2018(self):
        """Babusiaux, Gaia Collaboration et al. 2018, Table A1b (>250 pc):
        http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/616/A10
        """
        fp = join(
            self.data_loc,
            "TablesGaiaDR2HRDpaper/TableA1b_37 open cluster members beyond 250 pc.csv",
        )
        df = pd.read_csv(fp, delimiter=",", comment="#")
        df = df.replace(r"^\s*$", np.nan, regex=True)
        df.columns = [c.strip() for c in df.columns]
        df.Cluster = df.Cluster.apply(lambda x: x.strip())
        return df

    # def get_open_cluster_members_far_parallax(self, save_csv=True):
    #     """gaia ari provides plx values for d>250pc cluster members"""
    #
    #     df = get_open_cluster_members_far()
    #     plx = []
    #     for (r, d, m) in tqdm(df[["ra", "dec", "Gmag"]].values):
    #         coord = SkyCoord(ra=r, dec=d, unit="deg")
    #         g = Gaia.query_object(coord, radius=10 * u.arcsec).to_pandas()
    #         gcoords = SkyCoord(ra=g["ra"], dec=g["dec"], unit="deg")
    #         # FIXME: get minimum or a few stars around the minimum?
    #         idx = coord.separation(gcoords).argmin()
    #         if abs(m - g.loc[idx, "g_mean_mag"]) > 1.0:
    #             star = g.loc[idx]
    #         else:
    #             star = g.loc[1]
    #
    #         plx.append(star["parallax"].values[0])
    #     if save_csv:
    #         df["plx"] = plx
    #         fp = "open_cluster_members_far_parallax.csv"
    #         df_open_mem_concat.to_csv(join(dataloc, fp), index=False)
    #         print(f"Saved: {fp}")
    #     return df

    # def combine_open_clusters(dataloc="../data/", clobber=False, save_csv=False):
    #     """ Gaia DR2 open cluster + 269 clusters with ages"""
    #     fp = join(dataloc, "merged_open_clusters.csv")
    #     if not exists(fp) or clobber:
    #         df1 = tql.get_269_open_clusterss(dataloc=dataloc)
    #         df2 = tql.get_open_clusters()
    #         df2 = df2.rename(columns={"RA": "RA_ICRS", "Dec": "DE_ICRS"})
    #         df2.Cluster = df2.Cluster.apply(lambda x: x.replace(" ", ""))
    #         df_all = pd.merge(
    #             left=df1[
    #                 ["Cluster", "RA_ICRS", "DE_ICRS", "D_est[pc]", "Age[Myr]"]
    #             ],
    #             right=df2[
    #                 ["Cluster", "RA_ICRS", "DE_ICRS", "D_est[pc]", "Age[Myr]"]
    #             ],
    #             on="Cluster",
    #             how="outer",
    #         )
    #         print("\n=====Edit columns=====\n")
    #         if save_csv:
    #             df_all.to_csv(fp)
    #             print(f"Saved: {fp}")
    #     else:
    #         df_all = pd.read_csv(fp)
    #     return df_all
    #
    #
    # def combine_open_cluster_members_near_far(
    #     dataloc="../data/TablesGaiaDR2HRDpaper/", save_csv=False
    # ):
    #     # make uniform column
    #     df_open_near_mem = get_open_cluster_members_near(dataloc)
    #     df_open_far_mem = get_open_cluster_members_far(dataloc)
    #     # concatenate
    #     df_open_mem_concat = pd.concat(
    #         [df_open_near_mem, df_open_far_mem], sort=True, join="outer"
    #     )
    #     if save_csv:
    #         fp = "open_cluster_members.csv"
    #         df_open_mem_concat.to_csv(join(dataloc, fp), index=False)
    #         print(f"Saved: {fp}")
    #     return df_open_mem_concat
    #
    #
    # def combine_open_clusters_near_far(
    #     dataloc="../data/TablesGaiaDR2HRDpaper/", save_csv=False
    # ):
    #     # make uniform column
    #     df_open_near = get_open_clusters_near(dataloc)
    #     df_open_far = get_open_clusters_far(dataloc)
    #     df_open_far["RA_ICRS"] = df_open_far["RAJ2000"]
    #     df_open_far["DE_ICRS"] = df_open_far["DEJ2000"]
    #     df_open_far = df_open_far.drop(["RAJ2000", "DEJ2000"], axis=1)
    #
    #     # concatenate
    #     df_open_concat = pd.concat(
    #         [df_open_near, df_open_far], sort=True, join="outer"
    #     )
    #     if save_csv:
    #         fp = "open_clusters.csv"
    #         df_open_concat.to_csv(join(dataloc, fp), index=False)
    #         print(f"Saved: {fp}")
    #     return df_open_concat

    def get_clusters_Dias2002_2015(self):
        """Dias et al. 2004-2015; compiled until 2016:
        https://ui.adsabs.harvard.edu/abs/2014yCat....102022D/abstract
        """
        fp = join(
            self.data_loc,
            "TablesDias2014/2167_open_clusters_and_candidates.tsv",
        )
        df = pd.read_csv(fp, delimiter="\t", comment="#")
        coords = SkyCoord(
            ra=df["RAJ2000"].values,
            dec=df["DEJ2000"].values,
            unit=("hourangle", "deg"),
            equinox="J2000",
        )
        # replace space in cluster name with underscore (Bouma19 convention)
        df["Cluster"] = df.Cluster.apply(lambda x: "_".join(x.split()))
        df["RAJ2000"] = coords.ra.deg
        df["DEJ2000"] = coords.dec.deg
        # drop some columns
        df = df.drop(["P", "WEBDA", "Lynga"], axis=1)
        # rename columns to follow Bouma19 convention
        df = df.rename(
            columns={
                "RAJ2000": "ra",
                "DEJ2000": "dec",
                "Dist": "distance",
                "Diam": "ang_diameter",
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "K14": "details",
                "RV": "radial_velocity",
                "Age": "log10_age",
                "o_RV": "RV_nstars",
                "o_[Fe/H]": "[Fe/H]_nstars",
                "TrType": "TrumplerType",
            }
        )
        return df

    def get_clusters_Bossini2019(self):
        """Bossini et al. 2019:
        http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A%2BA/623/A108
        """
        fp = join(
            self.data_loc,
            "TablesBossini2019/Bossini2019_269_open_clusters.tsv",
        )
        df = pd.read_table(fp, delimiter="\t", skiprows=69, comment="#")
        # convert distance and age
        df["distance"] = Distance(distmod=df["Dist"]).pc
        # add underscore in cluster name (Bouma19 convention)
        regex = re.compile("([a-zA-Z]+)([0-9]+)")  # str and number separator
        df["Cluster"] = df.Cluster.apply(
            lambda x: "_".join(regex.match(x).groups())
        )
        # df[["Cluster", "log10_age", "distance"]].sort_values(by="log10_age")
        # rename columns
        df = df.rename(
            columns={
                "Dist": "dist_mod",
                "e_Dist": "dist_mod_e1",
                "logA": "log10_age",
                "E_Dist": "dist_mod_e2",
                "RA_ICRS": "ra",
                "DE_ICRS": "dec",
            }
        )
        return df

    def get_clusters_Karchenko2013(self):
        """Karchenko et al. 2013
        http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/558/A53
        """
        fp = join(
            self.data_loc,
            "TablesKarchenko2013/Catalog of parameters for open clusters.tsv",
        )
        df = pd.read_table(fp, delimiter="\t", skiprows=48, comment="#")
        df = df.rename(
            columns={
                "RAJ2000": "ra",
                "DEJ2000": "dec",
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "d": "distance",
                "logt": "log10_age",
                "RV": "radial_velocity",
                "r0": "ang_radius_core",
                "r1": "ang_radius_central",
                "r2": "ang_radius",
            }
        )
        # remove columns
        df = df.drop(["map", "cmd", "stars", "Simbad"], axis=1)
        return df


class Cluster(ClusterCatalog):
    def __init__(
        self,
        cluster_name,
        catalog_name="Bouma2019",
        data_loc=DATA_PATH,
        verbose=True,
    ):
        super().__init__(
            catalog_name=catalog_name, data_loc=data_loc, verbose=verbose
        )
        self.all_members = self.query_catalog(return_members=True)
        self.cluster_name = cluster_name
        self.cluster_members = None
        self.cluster_members_gaia_params = None
        self.verbose = verbose
        # self.cluster_summary = None
        errmsg = f"{self.cluster_name} is not found in {self.catalog_name}"
        assert np.any(
            self.all_members.Cluster.isin([self.cluster_name])
        ), errmsg

    def query_cluster_members(self):
        idx = self.all_members.Cluster.isin([self.cluster_name])
        df = self.all_members.loc[idx]
        self.cluster_members = df
        return df

    def query_cluster_members_gaia_params(
        self,
        df=None,
        radius=3,
        clobber=False,
        gmag_cut=None,
        top_n_brighest=None,
        data_loc=DATA_PATH,
    ):
        """query gaia params for each cluster member
        Parameters
        ----------
        df : pandas.DataFrame
            cluster catalog to match against
        radius : float
            search radius in arcsec
        top_n_brighest : int
            query only bright stars in cluster (recommended for members>300)
        gmag_cut : float
            query only for stars brighter than gmag_cut (recommended for members>300)

        Returns
        -------
        tab : pandas.DataFrame
            table of matches
        """
        # fp=join(data_loc,f'TablesGaiaDR2HRDpaper/{cluster_name}_members.hdf5')
        if self.cluster_members is None:
            self.query_cluster_members()
        if df is None:
            df = self.cluster_members

        fp = join(data_loc, f"{self.cluster_name}_members.hdf5")
        if not exists(fp) or clobber:
            gaia_data = {}
            if top_n_brighest is not None:
                # sort in decreasing magnitude
                df = df.sort_values("phot_g_mean_mag", ascending=False)
                # use only top_n_brighest
                gaiaids = df.iloc[top_n_brighest, "source_id"].values
            elif gmag_cut is not None:
                # apply Gmag cut
                idx = df["phot_g_mean_mag"] < gmag_cut
                gaiaids = df.loc[idx, "source_id"].values
            else:
                # use all ids
                gaiaids = df["source_id"].values

            if self.verbose:
                print(
                    f"Querying Gaia DR2 for {len(gaiaids)} {self.cluster_name} members.\n"
                )
            for gaiaid in tqdm(gaiaids):
                try:
                    errmsg = (
                        f"multiple cluster names found: {df.Cluster.unique()}"
                    )
                    assert np.all(df.Cluster.isin([self.cluster_name])), errmsg
                    # query gaia to populate target parameters including its distance
                    t = Target(gaiaDR2id=gaiaid, verbose=self.verbose)
                    df_gaia = t.query_gaia_dr2_catalog(
                        radius=radius, return_nearest_xmatch=True
                    )
                    gaia_data[gaiaid] = df_gaia
                except Exception as e:
                    print(e)
            # save
            dd.io.save(fp, gaia_data)
            msg = f"Saved: {fp}"
        else:
            gaia_data = dd.io.load(fp)
            msg = f"Loaded: {fp}"
        if self.verbose:
            print(msg)
        # convert dict of series into a single df
        sample = list(gaia_data.values())[0]  # any would do
        if isinstance(sample, pd.Series):
            df_gaia = pd.concat(gaia_data, axis=1, ignore_index=False).T
        # convert dict of df into a single df
        else:
            df_gaia = pd.concat(gaia_data.values(), ignore_index=True)

        if gmag_cut is not None:
            # apply Gmag cut
            idx = df_gaia["phot_g_mean_mag"] < gmag_cut
            df_gaia = df_gaia[idx]
        self.cluster_members_gaia_params = df_gaia
        return df_gaia

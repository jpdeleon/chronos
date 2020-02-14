# -*- coding: utf-8 -*-

r"""
classes for searching and querying cluster catalogs

See also from astroquery.xmatch import XMatch
"""

# Import standard library
from os.path import join, exists
import logging
import re

# Import modules
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, Distance
from astropy import units as u
from tqdm import tqdm
import deepdish as dd

# from lightkurve.search import _query_mast as query_mast
# Import from package
from chronos import target
from chronos.config import DATA_PATH

log = logging.getLogger(__name__)

__all__ = ["Cluster", "ClusterCatalog"]

CATALOG_LIST = [
    "Bouma2019",
    "Babusiaux2018",
    "CantatGaudin2018",
    "Bossini2019",
    "Dias2014",
    "Karchenko2013",
    "Cody2018",
]


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
                    t = target.Target(gaiaDR2id=gaiaid, verbose=self.verbose)
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

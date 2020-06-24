# -*- coding: utf-8 -*-

r"""
classes for searching and querying cluster catalogs

See also from astroquery.xmatch import XMatch
"""

# Import standard library
from pathlib import Path
from glob import glob
from pprint import pprint
from inspect import signature
import itertools
import logging
import re

# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib.patches import Ellipse
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord, Distance
from astropy.table import Table
import astropy.units as u
from tqdm import tqdm
import deepdish as dd

# Import from package
from chronos import target
from chronos.utils import (
    get_absolute_gmag,
    get_absolute_color_index,
    get_transformed_coord,
)
from chronos.config import DATA_PATH

log = logging.getLogger(__name__)

__all__ = [
    "Cluster",
    "ClusterCatalog",
    "CatalogDownloader",
    "plot_cmd",
    "plot_hrd",
    "plot_rdp_pmrv",
    "plot_xyz_uvw",
    "plot_xyz_3d",
]

CATALOG_DICT = {
    # 570 new open clusters in the Galactic disc
    "CastroGinard2020": "J/A+A/635/A45",
    # 1481 clusters and their members
    "CantatGaudin2020": "J/A+A/633/A99",
    # open clusters in the Galactic anticenter
    "CastroGinard2019": "J/A+A/627/A35",
    #
    "CantatGaudin2018": "J/A+A/618/A93",
    # HRD of Gaia DR2
    "Babusiaux2018": "J/A+A/616/A10",
    # merged catalogs
    "Bouma2019": "J/ApJS/245/13",
    # Banyan sigma
    "Gagne2018a": "J/ApJ/860/43",  # TGAS
    "Gagne2018b": "J/ApJ/862/138",  # DR2
    # Argus assoc via simbad link
    "Zuckerman2019": "None",
    # eta Cha assoc
    "Murphy2013": "J/MNRAS/435/1325",
    # nu Cha assoc
    "Bell2015": "J/MNRAS/454/593",
    # ages of 269 OC
    "Bossini2019": "J/A+A/623/A108/tablea",
    # OC
    "Sampedro2017": "J/MNRAS/470/3937",
    "Randich2018": "J/A+A/612/A99",
    "Karchenko2013": "J/A+A/558/A53",
    # OC #"Dias2014"?
    "Dias2016": "B/ocl",
    # Psc Eri
    "Curtis2019": "J/AJ/158/77",
    # praesepe, alpa per
    "Lodieu2019": "J/A+A/628/A66",
    # young RV
    # 'Schneider2019': 'J/AJ/157/234',
    # young harps RV
    "Grandjean2020": "J/A+A/633/A44",
    # 'Carerra2019': 'J/A+A/623/A80', #apogee+galah
    # 'BailerJones2018': 'I/347', #distances
    # 'Luo2019': 'V/149', #Lamost
    # 'Olivares2019': 'J/A+A/625/A115', #Ruprecht 147 DANCe: oldest open cluster @ 300pc
    # "Cody2018": "",
}

CATALOG_LIST = [key for key in CATALOG_DICT.keys()]


class CatalogDownloader:
    """download tables from vizier

    Attributes
    ----------
    tables : astroquery.utils.TableList
        collection of astropy.table.Table downloaded from vizier
    """

    def __init__(
        self, catalog_name, data_loc=DATA_PATH, verbose=True, clobber=False
    ):
        self.catalog_name = catalog_name
        self.catalog_dict = CATALOG_DICT
        self.verbose = verbose
        self.clobber = clobber
        self.data_loc = Path(data_loc, self.catalog_name)
        self.tables = None

    def get_tables_from_vizier(self, row_limit=50, save=False, clobber=None):
        """row_limit-1 to download all rows"""
        clobber = self.clobber if clobber is None else clobber
        if row_limit == -1:
            msg = f"Downloading all tables in "
        else:
            msg = f"Downloading the first {row_limit} rows of each table "
        msg += f"{self.catalog_dict[self.catalog_name]} from vizier."
        if self.verbose:
            print(msg)
        # set row limit
        Vizier.ROW_LIMIT = row_limit

        tables = Vizier.get_catalogs(self.catalog_dict[self.catalog_name])
        errmsg = f"No data returned from Vizier."
        assert tables is not None, errmsg
        self.tables = tables

        if self.verbose:
            pprint({k: tables[k]._meta["description"] for k in tables.keys()})

        if save:
            self.save_tables(clobber=clobber)
        return tables

    def save_tables(self, clobber=None):
        errmsg = "No tables to save"
        assert self.tables is not None, errmsg
        clobber = self.clobber if clobber is None else clobber

        if not self.data_loc.exists():
            self.data_loc.mkdir()

        for n, table in enumerate(self.tables):
            fp = Path(self.data_loc, f"{self.catalog_name}_tab{n}.txt")
            if not fp.exists() or clobber:
                table.write(fp, format="ascii")
                if self.verbose:
                    print(f"Saved: {fp}")
            else:
                print("Set clobber=True to overwrite.")

    def get_vizier_url(self, catalog_name=None):
        if catalog_name is None:
            catalog_name = self.catalog_name
        base_url = "https://vizier.u-strasbg.fr/viz-bin/VizieR?-source="
        vizier_key = self.catalog_dict[catalog_name]
        return base_url + vizier_key

    def __repr__(self):
        """Override to print a readable string representation of class
        """
        included_args = ["catalog_name", "cluster_name"]
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


class ClusterCatalog(CatalogDownloader):
    # __slots__ = ["catalog_name", "all_clusters", "all_members", "catalog_list",
    # "verbose", "clobber", "data_loc"
    # ]
    def __init__(
        self,
        catalog_name="CantatGaudin2020",
        verbose=True,
        clobber=False,
        data_loc=DATA_PATH,
    ):
        super().__init__(
            catalog_name=catalog_name,
            data_loc=data_loc,
            verbose=verbose,
            clobber=clobber,
        )
        """Initialize the catalog

        Attributes
        ----------
        data_loc : str
            data directory
        all_members: pd.DataFrame
            list of all members in catalog
        all_clusters : pd.DataFrame
            list of all clusters in catalog

        Note:
        setting `all_members` as a method (as opposed to attribute)
        seems not
        """
        self.catalog_list = CATALOG_LIST
        self.all_clusters = None  # self.query_catalog(return_members=False)
        self.all_members = None

        # files = glob(join(self.data_loc, "*.txt"))
        if self.data_loc.exists():  # & len(files)<2:
            if self.clobber:
                _ = self.get_tables_from_vizier(
                    row_limit=-1, save=True, clobber=self.clobber
                )
        else:
            _ = self.get_tables_from_vizier(
                row_limit=-1, save=True, clobber=self.clobber
            )

    def query_catalog(self, name=None, return_members=False, **kwargs):
        """Query catalogs

        Parameters
        ----------
        name : str
            catalog name; see `self.catalog_list`
        return_members : bool
            return parameters for all members instead of the default

        Returns
        -------
        df : pandas.DataFrame
            dataframe parameters of the cluster or its individual members
        Note:
        1. See self.vizier_url() for details
        2. Use the following:
        if np.any(df["parallax"] < 0):
            df = df[(df["parallax"] >= 0) | (df["parallax"].isnull())]
            if verbose:
                print("Some parallaxes are negative!")
                print("These are removed for the meantime.")
                print("For proper treatment, see:")
                print("https://arxiv.org/pdf/1804.09366.pdf\n")

        FIXME: self.all_clusters and self.all_members are repeated each if else block
        """
        self.catalog_name = name if name is not None else self.catalog_name
        if self.verbose:
            print(f"Using {self.catalog_name} catalog.")
        if self.catalog_name == "Bouma2019":
            if return_members:
                df_mem = self.get_members_Bouma2019()
                self.all_members = df_mem
                return df_mem
            else:
                df = self.get_clusters_Bouma2019()
                self.all_clusters = df
                return df
        elif self.catalog_name == "CantatGaudin2020":
            if return_members:
                df_mem = self.get_members_CantatGaudin2020()
                self.all_members = df_mem
                return df_mem
            else:
                df = self.get_clusters_CantatGaudin2020()
                self.all_clusters = df
                return df
        elif self.catalog_name == "CastroGinard2020":
            if return_members:
                df_mem = self.get_members_CastroGinard2020()
                self.all_members = df_mem
                return df_mem
            else:
                df = self.get_clusters_CastroGinard2020()
                self.all_clusters = df
                return df
        elif self.catalog_name == "CastroGinard2019":
            if return_members:
                df_mem = self.get_members_CastroGinard2019()
                self.all_members = df_mem
                return df_mem
            else:
                df = self.get_clusters_CastroGinard2019()
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
        elif self.catalog_name == "Babusiaux2018":
            if return_members:
                df_mem = self.get_members_Babusiaux2018()
                # raise NotImplementedError("To be updated")
                # return self.get_members_Babusiaux2018_near() #fewer members
                self.all_members = df_mem
                return df_mem  # has parallaxes
            else:
                df = self.get_clusters_Babusiaux2018()
                self.all_clusters = df
                return df
        elif self.catalog_name == "Gagne2018a":
            print("Not cluster but young, moving group catalog")
            if return_members:
                df_mem = self.get_members_Gagne2018a()
                self.all_members = df_mem
                return df_mem
            else:
                df = self.get_clusters_Gagne2018a()
                self.all_clusters = df
                return df
        elif self.catalog_name == "Gagne2018b":
            print("Not cluster but young, moving group catalog")
            if return_members:
                df_mem = self.get_members_Gagne2018b()
                self.all_members = df_mem
                return df_mem
            else:
                df = self.get_clusters_Gagne2018b()
                self.all_clusters = df
                return df
        elif self.catalog_name == "Zuckerman2019":
            print("Not cluster but only Argus association")
            df_mem = self.get_members_Zuckerman2019()
            self.all_members = df_mem
            return df_mem
        elif self.catalog_name == "Murphy2013":
            print("Not cluster but only eta Cha association")
            if return_members:
                df_mem = self.get_members_Murphy2013()
                self.all_members = df_mem
                return df_mem
            else:
                df = self.get_clusters_Murphy2013()
                self.all_clusters = df
                return df
        elif self.catalog_name == "Bell2015":
            print("Not cluster but only nu Cha association")
            if return_members:
                df_mem = self.get_members_Bell2015()
                self.all_members = df_mem
                return df_mem
            else:
                df = self.get_clusters_Bell2015()
                self.all_clusters = df
                return df
        elif self.catalog_name == "Dias2016":
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
                    "No individual cluster members in Bossini catalog"
                )
            else:
                df = self.get_clusters_Bossini2019()
                self.all_clusters = df
                return df
        elif self.catalog_name == "Sampedro2017":
            if return_members:
                df_mem = self.get_members_Sampedro2017()
                self.all_members = df_mem
                return df_mem
            else:
                df = self.get_clusters_Sampedro2017()
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
        elif self.catalog_name == "Curtis2019":
            if return_members:
                raise ValueError(
                    "No individual cluster members in Curtis2019 catalog"
                )
            else:
                df = self.get_clusters_Curtis2019()
                self.all_clusters = df
                return df
        elif self.catalog_name == "Lodieu2019":
            if return_members:
                df_mem = self.get_members_Lodieu2019()
                self.all_members = df_mem
                return df_mem
            else:
                raise ValueError(
                    "No individual cluster info in Lodieu2019 catalog"
                )
        elif self.catalog_name == "Grandjean2020":
            # return both cluster and members
            print("Not cluster but young, nearby RV catalog")
            if return_members:
                df = self.get_catalog_Grandjean2020()
                self.all_members = df
                return df
            else:
                df = self.get_catalog_Grandjean2020()
                self.all_clusters = df
                return df
        elif self.catalog_name in self.catalog_list:
            raise NotImplementedError("Catalog to be added later.")
        # add more catalogs here
        else:
            raise ValueError(
                f"Catalog name not found in list: {self.catalog_list}"
            )

    def get_clusters_CastroGinard2020(self):
        """Castro-Ginard et al. 2020,
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab0.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        df = df.applymap(
            lambda x: x.decode("ascii") if isinstance(x, bytes) else x
        )
        df = df.rename(
            columns={
                "RA_ICRS": "raJ2015",
                "e_RA_ICRS": "e_raJ2015",
                "DE_ICRS": "decJ2015",
                "e_DE_ICRS": "e_decJ2015",
                "_RA.icrs": "ra",
                "_DE.icrs": "dec",
                "plx": "parallax",
                "e_plx": "e_parallax",
                "pmRA": "pmra",
                "e_pmRA": "e_pmra",
                "pmDE": "pmdec",
                "e_pmDE": "e_pmdec",
            }
        )
        # add distance
        df["distance"] = Distance(parallax=df.parallax.values * u.mas).pc
        return df

    def get_members_CastroGinard2020(self):
        """Castro-Ginard et al. 2020,
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab1.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        df = df.applymap(
            lambda x: x.decode("ascii") if isinstance(x, bytes) else x
        )
        df = df.rename(
            columns={
                "RA_ICRS": "raJ2015",
                "DE_ICRS": "decJ2015",
                "_RA.icrs": "ra",
                "_DE.icrs": "dec",
                "plx": "parallax",
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "Source": "source_id",
            }
        )
        return df

    def get_clusters_CastroGinard2019(self):
        """Castro-Ginard et al. 2019,
        open clusters in the galactic anti-center
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab0.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        df = df.applymap(
            lambda x: x.decode("ascii") if isinstance(x, bytes) else x
        )
        df = df.rename(
            columns={
                "RA_ICRS": "raJ2015",
                "e_RA_ICRS": "e_raJ2015",
                "DE_ICRS": "decJ2015",
                "e_DE_ICRS": "e_decJ2015",
                "_RA.icrs": "ra",
                "_DE.icrs": "dec",
                "Plx": "parallax",
                "e_plx": "e_parallax",
                "pmRA": "pmra",
                "e_pmRA": "e_pmra",
                "pmDE": "pmdec",
                "e_pmDE": "e_pmdec",
            }
        )
        # add distance
        df["distance"] = Distance(parallax=df.parallax.values * u.mas).pc
        return df

    def get_members_CastroGinard2019(self):
        """Castro-Ginard et al. 2019,
        open clusters in the galactic anti-center
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab1.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        df = df.applymap(
            lambda x: x.decode("ascii") if isinstance(x, bytes) else x
        )
        df = df.rename(
            columns={
                "RA_ICRS": "raJ2015",
                "DE_ICRS": "decJ2015",
                "_RA.icrs": "ra",
                "_DE.icrs": "dec",
                "Plx": "parallax",
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "Source": "source_id",
            }
        )
        return df

    def get_members_Bouma2019(self):
        """
        Bouma et al. 2019:
        https://ui.adsabs.harvard.edu/abs/2019arXiv191001133B/abstract

        Source:
        iopscience.iop.org/0067-0049/245/1/13/suppdata/apjsab4a7et1_mrt.txt

        To produce master_list.csv used here, the original file is read into memory
        using `astropy.io.ascii.read` with specified formatting, then turned into
        `pd.DataFrame` and saved as csv.
        """
        fp = Path(self.data_loc, "OC_MG_FINAL_v0.3_publishable.csv")
        df = pd.read_csv(fp, header=0, sep=";")
        df = df.rename(
            columns={"cluster": "clusters", "unique_cluster_name": "Cluster"}
        )
        # fp = Path(self.data_loc, "TablesBouma2019/master_list.csv")
        # df = pd.read_csv(fp)
        # df = df.rename(
        #     columns={
        #         "Cluster": "clusters",
        #         "Ref": "reference",
        #         "CName": "ext_catalog_name",
        #         "RAdeg": "ra",
        #         "DEdeg": "dec",
        #         "pmRA": "pmra",
        #         "pmDE": "pmdec",
        #         "plx": "parallax",
        #         "Gmag": "phot_g_mean_mag",
        #         "GBp": "phot_bp_mean_mag",
        #         "GRp": "phot_rp_mean_mag",
        #         "K13": "k13_name_match",
        #         "Unique": "Cluster",
        #         "How": "how_match",
        #         "inK13": "in_k13",
        #         "Com": "comment",
        #         "logt": "k13_logt",
        #         "e_logt": "k13_e_logt",
        #     }
        # )
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
            equinox="J2000.0",  # or 2015.5?
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

    def get_clusters_CantatGaudin2020(self):
        """Cantat-Gaudin et al. 2020:
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab0.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        df = _decode_n_drop(df, ["SimbadName", "dmode_01", "dmode-01"])
        df = df.rename(
            columns={
                "RA_ICRS": "raJ2015",
                "DE_ICRS": "decJ2015",
                "_RA.icrs": "ra",
                "_DE.icrs": "dec",
                "Plx": "parallax",
                "dmode": "distance",
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "N": "Nstars",
            }
        )
        return df

    def get_members_CantatGaudin2020(self):
        """Cantat-Gaudin et al. 2020:
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab1.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        df = df.applymap(
            lambda x: x.decode("ascii") if isinstance(x, bytes) else x
        )
        df = df.rename(
            columns={
                "RA_ICRS": "raJ2015",
                "DE_ICRS": "decJ2015",
                "_RA.icrs": "ra",
                "_DE.icrs": "dec",
                "Source": "source_id",  # 'RV':'radial_velocity',
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "Plx": "parallax",
                "Gmag": "phot_g_mean_mag",
                "BP-RP": "bp_rp",
            }
        )
        return df

    def get_clusters_CantatGaudin2018(self):
        """Cantat-Gaudin et al. 2018:
        http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/618/A93
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab0.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        df = _decode_n_drop(df, ["SimbadName", "dmode_01", "dmode-01"])
        df = df.rename(
            columns={
                "RAJ2000": "ra",
                "DEJ2000": "dec",
                "dmode": "distance",
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "plx": "parallax",
            }
        )
        return df

    def get_members_CantatGaudin2018(self):
        """Cantat-Gaudin et al. 2018:
        http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/618/A93
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab1.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        df = df.applymap(
            lambda x: x.decode("ascii") if isinstance(x, bytes) else x
        )
        df = df.rename(
            columns={
                "RA_ICRS": "raJ2015",
                "DE_ICRS": "decJ2015",
                "_RA.icrs": "ra",
                "_DE.icrs": "dec",
                "Source": "source_id",
                "o_Gmag": "phot_g_n_obs",
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "plx": "parallax",
                # 'Gmag': 'phot_g_mean_mag',
                "BP-RP": "bp_rp",
            }
        )
        return df

    def get_clusters_Babusiaux2018(self):
        """Babusiaux, Gaia Collaboration et al. 2018
        Table 3 (<250 pc) & Table 4 (>250 pc):
        http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/616/A10
        """
        fp1 = Path(self.data_loc, f"{self.catalog_name}_tab2.txt")
        tab1 = Table.read(fp1, format="ascii").to_pandas()
        tab1 = tab1.rename(
            columns={"_RA.icrs": "RAJ2000", "_DE.icrs": "DEJ2000"}
        )
        fp2 = Path(self.data_loc, f"{self.catalog_name}_tab3.txt")
        tab2 = Table.read(fp2, format="ascii").to_pandas()
        df = pd.concat([tab1, tab2], axis=0, join="outer")
        df = _decode_n_drop(df, ["SimbadName"])
        df = df.rename(
            columns={
                "RA_ICRS": "raJ2015",
                "DE_ICRS": "decJ2015",
                # '_RA.icrs':'ra', '_DE.icrs':'dec',
                "RAJ2000": "ra",
                "DEJ2000": "dec",
                # 'RV':'radial_velocity', 'e_RV':'e_radial_velocity',
                "o_RV": "RV_n_obs",
                "NMemb": "Nstars",
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "plx": "parallax",
            }
        )
        df["distance"] = Distance(parallax=df["parallax"].values * u.mas).pc
        return df

    def get_members_Babusiaux2018(self):
        """Babusiaux, Gaia Collaboration et al. 2018,
        Table A1a (<250 pc) & Table A1b (>250 pc):
        http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/616/A10
        """
        fp1 = Path(self.data_loc, f"{self.catalog_name}_tab0.txt")
        tab1 = Table.read(fp1, format="ascii").to_pandas()
        fp2 = Path(self.data_loc, f"{self.catalog_name}_tab1.txt")
        tab2 = Table.read(fp2, format="ascii").to_pandas()
        df = pd.concat([tab1, tab2], axis=0, join="outer")
        df = _decode_n_drop(df, ["Simbad"])
        df = df.applymap(
            lambda x: x.decode("ascii") if isinstance(x, bytes) else x
        )
        df = df.rename(
            columns={
                "RAdeg": "raJ2015",
                "DEdeg": "decJ2015",
                "Source": "source_id",
                "_RA": "ra",
                "_DE": "dec",
                "plx": "parallax",
            }
        )
        return df

    def get_catalog_Grandjean2020(self):
        """Grandjean et al. 2020,
        'J/A+A/633/A44/sources': 'List of targets and positions',
        'J/A+A/633/A44/tablea1': 'Stellar characteristics of the survey',
        'J/A+A/633/A44/tablea2': 'Results of the survey'
        """
        dfs = []
        for i in range(3):
            fp = Path(self.data_loc, f"{self.catalog_name}_tab{i}.txt")
            tab = Table.read(fp, format="ascii")
            df = tab.to_pandas()
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        # remove duplicate Name columns
        df = df.loc[:, ~df.columns.duplicated()]
        df = _decode_n_drop(df, ["Simbad"])
        coords = SkyCoord(
            ra=df["RAJ2000"], dec=df["DEJ2000"], unit=("hourangle", "degree")
        )
        df["ra"] = coords.ra.deg
        df["dec"] = coords.dec.deg
        df = df.drop(["RAJ2000", "DEJ2000"], axis=1)
        return df

    def get_clusters_Gagne2018a(self):
        """
        BANYAN. XII. New members from Gaia-Tycho data (Gagne+, 2018a)
        https://ui.adsabs.harvard.edu/abs/2018ApJ...860...43G/abstract

        J/ApJ/860/43/refs	References (table added by CDS) (148 rows)
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab5.txt")
        df = Table.read(fp, format="ascii").to_pandas()
        return df

    def get_members_Gagne2018a(self):
        """
        BANYAN XII
        https://ui.adsabs.harvard.edu/abs/2018ApJ...862..138G/abstract
        J/ApJ/860/43/table4 (c)Candidate members recovered in this work (708 rows)
        J/ApJ/860/43/seq1	*CMD for the bona fide members of young assoc – Ages<20Myr (100 rows)
        J/ApJ/860/43/seq2	*CMD for the bona fide members of young assoc – Ages: 20-100Myr (100 rows)
        J/ApJ/860/43/seq3	*CMD for the bona fide members of young assoc – Ages: 100-800Myr (100 rows)
        J/ApJ/860/43/seq4	*CMD – Ages: Field (>800Myr) (Interactive plot) (Note) (100 rows)
        """
        fp0 = Path(self.data_loc, f"{self.catalog_name}_tab0.txt")
        tab0 = Table.read(fp0, format="ascii").to_pandas()
        # fp1 = Path(self.data_loc, f"{self.catalog_name}_tab1.txt")
        # tab1 = Table.read(fp1, format="ascii").to_pandas()
        # fp2 = Path(self.data_loc, f"{self.catalog_name}_tab2.txt")
        # tab2 = Table.read(fp2, format="ascii").to_pandas()
        # fp3 = Path(self.data_loc, f"{self.catalog_name}_tab3.txt")
        # tab3 = Table.read(fp3, format="ascii").to_pandas()
        # fp4 = Path(self.data_loc, f"{self.catalog_name}_tab4.txt")
        # tab4 = Table.read(fp4, format="ascii").to_pandas()
        # df = pd.concat([tab0,tab1,tab2,tab3,tab4], axis=0, join="outer")).reset_index()
        df = _decode_n_drop(tab0, ["SimbadName", "Gaia", "_2M"])
        df = df.rename(
            columns={
                "Assoc": "Cluster",
                "RAJ2000": "ra",
                "DEJ2000": "dec",
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "plx": "parallax",
                "RVel": "RV",
                "Dist": "distance",
            }
        )
        return df

    def get_clusters_Gagne2018b(self):
        """
        BANYAN XIII. A First Look at Nearby Young Associations with Gaia DR2
        https://ui.adsabs.harvard.edu/abs/2018ApJ...862..138G/abstract

        'J/ApJ/862/138/table1': 'Nearby young associations considered here',
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab0.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        df = _decode_n_drop(df, ["SimbadName"])
        df = df.rename(
            columns={
                "Assoc": "Cluster",
                "RAJ2000": "ra",
                "DEJ2000": "dec",
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "plx": "parallax",
                "RVel": "RV",
            }
        )
        return df

    def get_members_Gagne2018b(self):
        """
        BANYAN XIII. A First Look at Nearby Young Associations with Gaia DR2
        https://ui.adsabs.harvard.edu/abs/2018ApJ...862..138G/abstract

        'J/ApJ/862/138/table2': 'New candidates identified in this work',
        'J/ApJ/862/138/table3': 'Co-moving systems identified in this work',
        'J/ApJ/862/138/table5': 'New bona fide members'
        'J/ApJ/862/138/refs': 'References',
        """
        fp1 = Path(self.data_loc, f"{self.catalog_name}_tab1.txt")
        tab1 = Table.read(fp1, format="ascii").to_pandas()
        fp2 = Path(self.data_loc, f"{self.catalog_name}_tab2.txt")
        tab2 = Table.read(fp2, format="ascii").to_pandas()
        fp3 = Path(self.data_loc, f"{self.catalog_name}_tab3.txt")
        tab3 = Table.read(fp3, format="ascii").to_pandas()
        # fp4 = Path(self.data_loc, f"{self.catalog_name}_tab4.txt")
        # tab4 = Table.read(fp4, format="ascii").to_pandas()
        # tab4["Assoc"] = 'Upper Cr?'
        df = pd.concat([tab1, tab2, tab3], axis=0, join="outer").reset_index()
        df = _decode_n_drop(df, ["index", "GaiaDR2", "Simbad"])
        df = df.rename(
            columns={
                "Assoc": "Cluster",
                "RAJ2000": "ra",
                "DEJ2000": "dec",
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "plx": "parallax",
                "RVel": "RV",
                "r_RVel": "r_RV",
                "Gaia": "source_id",
            }
        )
        return df

    def get_members_Zuckerman2019(self):
        """Argus assoc
        https://ui.adsabs.harvard.edu/abs/2019ApJ...870...27Z/abstract
        """
        simbad_url = "http://simbad.u-strasbg.fr/simbad/sim-ref"
        simbad_url += "?querymethod=bib&simbo=on&submit=submit"
        simbad_url += "+bibcode&bibcode=2019ApJ...870...27Z"

        d = pd.read_html(simbad_url)
        return d[3]

    def get_clusters_Murphy2013(self):
        """Murphy+2013:
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab2.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        return df

    def get_members_Murphy2013(self):
        """Murphy+2013:
        """
        fp1 = Path(self.data_loc, f"{self.catalog_name}_tab0.txt")
        tab1 = Table.read(fp1, format="ascii").to_pandas()
        fp2 = Path(self.data_loc, f"{self.catalog_name}_tab1.txt")
        tab2 = Table.read(fp2, format="ascii").to_pandas()
        df = pd.concat([tab1, tab2], axis=0, join="outer")
        df = _decode_n_drop(df, ["Simbad"])
        df = df.rename(
            columns={
                "RAJ2000": "ra",
                "DEJ2000": "dec",
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "ID": "epsCha_id",
                "logTe": "logTeff",
                "logLb": "logLbol",
                "Dist": "distance",
            }
        )
        return df

    def get_clusters_Bell2015(self):
        """Bell+2015:
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab1.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        return df

    def get_members_Bell2015(self):
        """Bell+2015:
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab0.txt")
        df = Table.read(fp, format="ascii").to_pandas()
        df = df.rename(columns={"_RA": "ra", "_DE": "dec", "Dist": "distance"})
        return df

    def get_clusters_Sampedro2017(self):
        """
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab0.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        df = _decode_n_drop(df, ["SimbadName", "File"])
        df = df.rename(
            columns={
                "Name": "Cluster",
                "RAJ2000": "ra",
                "DEJ2000": "dec",
                "Dist": "distance",
                "logAge": "log10_age",
                "E_B-V_": "E_B-V",
                "plx": "parallax",
            }
        )
        return df

    def get_members_Sampedro2017(self):
        """
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab1.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        df = _decode_n_drop(df, ["GaiaDR2"])
        df = df.rename(
            columns={
                "RA_ICRS": "ra",
                "DE_ICRS": "dec",
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "gmag": "Gmag",
            }
        )
        return df

    def get_clusters_Dias2002_2015(self):
        """Dias et al. 2004-2015; compiled until 2016:
        https://ui.adsabs.harvard.edu/abs/2014yCat....102022D/abstract
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab0.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        df = _decode_n_drop(df, ["WEBDA", "Lynga"])
        df = df.rename(
            columns={
                "RAJ2000": "ra",
                "DEJ2000": "dec",
                "Dist": "distance",
                "Diam": "ang_diameter",
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "K14": "details",
                "Age": "log10_age",
                "o_RV": "RV_obs_n",
                "o_[Fe/H]": "[Fe/H]_obs_n",
                "TrType": "TrumplerType",
            }
        )
        return df

    def get_clusters_Bossini2019(self):
        """Bossini et al. 2019:
        http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A%2BA/623/A108
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab0.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        df = _decode_n_drop(df, ["SimbadName", "_RA.icrs", "_DE.icrs"])
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
        fp = Path(self.data_loc, f"{self.catalog_name}_tab0.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        df = _decode_n_drop(df, ["map", "cmd", "stars", "Simbad"])
        df = df.rename(
            columns={
                "RAJ2000": "ra",
                "DEJ2000": "dec",
                "pmRA": "pmra",
                "pmDE": "pmdec",
                "d": "distance",
                "logt": "log10_age",
                "r0": "ang_radius_core",
                "r1": "ang_radius_central",
                "r2": "ang_radius",
                "__Fe_H_": "Fe/H",
            }
        )
        return df

    def get_clusters_Curtis2019(self):
        """
        """
        fp = Path(self.data_loc, f"{self.catalog_name}_tab0.txt")
        tab = Table.read(fp, format="ascii")
        df = tab.to_pandas()
        df = _decode_n_drop(df, ["Seq", "Simbad"])
        df = df.rename(
            columns={
                "RA_ICRS": "ra",
                "DE_ICRS": "dec",
                "Source": "source_id",
                "GBP-GRP": "bp_rp",
            }
        )
        return df

    def get_members_Lodieu2019(self):
        """
        """
        dfs = []
        for n, name in enumerate(["alpha_per", "pleiades", "praesepe"]):
            fp = Path(self.data_loc, f"{self.catalog_name}_tab{n}.txt")
            tab = Table.read(fp, format="ascii")
            df = tab.to_pandas()
            # df = _decode_n_drop(df, ["Seq", "Simbad"])
            df = df.rename(
                columns={
                    "RA_ICRS": "ra",
                    "DE_ICRS": "dec",
                    "Source": "source_id",
                }
            )
            df["Cluster"] = name  # df.assign('Cluster', name)
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        return df

    def is_gaiaid_in_catalog(self):
        df_mem = self.query_catalog(return_members=True)
        if df_mem.source_id.isin([self.gaiaid]).sum() > 0:
            return True
        else:
            return False

    def plot_all_clusters(
        self, x=None, y=None, c=None, cmap=pl.get_cmap("Blues")
    ):
        """visualize all clusters in catalog"""
        n = "Cluster"
        if self.all_clusters is None:
            df = self.get_all_clusters()
        else:
            df = self.all_clusters
        _print_warning(df)

        def annotate_df(row):
            ax.annotate(
                row[n],
                row[[x, y]],
                xytext=(10, -5),
                textcoords="offset points",
                size=12,
                color="darkslategrey",
            )

        if (x is None) and (y is None):
            print(f"Choose `x` & `y` from\n{df.columns.tolist()}")
        else:
            ax = df.plot(kind="scatter", x=x, y=y, c=c, cmap=cmap)
            _ = df[[n, x, y]].apply(annotate_df, axis=1)
            ax.set_title(f"{self.catalog_name} catalog")
            return ax

    # @property
    # def all_clusters(self):
    #     return self.query_catalog(return_members=False)
    #
    # @property
    # def all_members(self):
    #     return self.query_catalog(return_members=True)

    def __call__(self):
        # should return all_clusters immediately?
        pass


class Cluster(ClusterCatalog):
    """
    Attributes
    ----------
    cluster_members : pd.DataFrame
        members of the given cluster

    Note:
    all_members attribute inherited from parent class
    is cleared from memory at the end of __init__ here

    FIXME: read only portion of astropy.Table
    so not all cluster members are always loaded to memory
    """

    def __init__(
        self,
        cluster_name,
        catalog_name="CantatGaudin2020",
        mission="tess",
        data_loc=DATA_PATH,
        verbose=True,
        clobber=False,
    ):
        super().__init__(
            catalog_name=catalog_name,
            data_loc=data_loc,
            verbose=verbose,
            clobber=clobber,
        )
        self.mission = mission
        self.cluster_name = cluster_name

        _ = self.query_catalog(return_members=True)
        idx = self.all_members.Cluster.isin([self.cluster_name])
        self.cluster_members = self.all_members.loc[idx].copy()
        # also self.all_members.query("Cluster==@self.cluster_name")

        all_cluster_names = list(self.all_members.Cluster.unique())
        errmsg = f"{self.cluster_name} is not found in {self.catalog_name}:\n"
        errmsg += f"{all_cluster_names}"
        assert np.any(
            self.all_members.Cluster.isin([self.cluster_name])
        ), errmsg
        self.cluster_members_gaia_params = None
        # Clear memory
        self.all_members = None

    def get_cluster_age(self, cluster_name=None):
        """
        log10(age) is taken from Bossini+2019 catalog
        """
        cluster_name = (
            self.cluster_name if cluster_name is None else cluster_name
        )
        cluster_name = cluster_name.lower().replace("_", "").strip()
        cc = ClusterCatalog(catalog_name="Bossini2019")
        cat = cc.query_catalog()
        cat.Cluster = cat.Cluster.apply(
            lambda x: x.lower().replace("_", "").strip()
        )
        idx = cat.Cluster.str.contains(cluster_name)
        if sum(idx) > 0:
            d = cat.loc[idx, ["Cluster", "log10_age"]]
            log10age = d.log10_age.values[0]
            print(
                f"log10(age)={log10age:.2f} yr = {(10**log10age)/1e6:.2f} Myr"
            )
            return log10age

    def query_cluster_members(self):
        """
        Note: this just method is defined just to follow the api in ClusterCatalog;
        cluster_members is just forwarded since it is
        already initializated in __init__
        """
        errmsg = "Re-initialize `Cluster` class"
        assert self.cluster_members is not None, errmsg
        return self.cluster_members

    def is_gaiaid_in_cluster(self):
        if self.cluster_members.source_id.isin([self.gaiaid]).sum() > 0:
            return True
        else:
            return False

    def get_nearest_cluster(self, coord, use_3d=True):
        if self.all_clusters is None:
            cat = self.query_catalog()
        else:
            cat = self.all_clusters
        errmsg = f"{self.catalog_name} does not include `distance`"
        assert "distance" in cat.columns, errmsg
        coords = SkyCoord(
            ra=cat["ra"],
            dec=cat["dec"],
            distance=cat["distance"],
            unit=("deg", "deg", "pc"),
        )
        if use_3d:
            sep = coords.separation_3d(coord)
        else:
            sep = coords.separation(coord)
        idx = sep.argmin()
        return cat.iloc[idx], sep[idx]

    def get_nearest_cluster_member(self, coord, use_3d=False):
        if self.all_members is None:
            mem = self.query_catalog(return_members=True)
        else:
            mem = self.all_members

        coords = SkyCoord(
            ra=mem["ra"],
            dec=mem["dec"],
            distance=mem["distance"],
            unit=("deg", "deg", "pc"),
        )
        if use_3d:
            sep = coords.separation_3d(coord)
        else:
            sep = coords.separation(coord)
        idx = sep.argmin()
        return mem.iloc[idx], sep[idx]

    def query_members_in_TGv8_catalog(self):
        """
        cross-match cluster members with TGv8 catalog
        which offers metallicity information, useful
        for galactic archaeology

        See Carillo2020: https://arxiv.org/abs/1911.07825
        """
        df = get_TGv8_catalog()
        idx = df.Gaia_source_id.isin(self.all_members.source_id)
        return df[idx]

    def plot_Toomre_diagrams(self, color="Fe_H"):
        """Toomre diagrams using [Fe/H] or [alpha/H] from
        APOGEE, RAVE, GALAH, LAMOST, Deacon, & Casagrande

        See Fig. 9 in https://arxiv.org/pdf/1911.07825.pdf
        Parameters
        ----------
        color : str
            stellar parameter used in color-coding scatter plot (default=Fe_H)
        """
        d = self.query_members_in_TGv8_catalog()
        cols = d.columns[d.columns.str.contains(color)]
        if d[cols].dropna(how="all", axis=1).shape[1]==0:
            raise ValueError(f"No data for {color}")
        else:
            fig, axs = pl.subplots(2, 3, figsize=(15,10),
                                   constrained_layout=True,
                                   sharex=True, sharey=True
                                  )
            ax = axs.flatten()

            catalogs = "APOGEE RAVE GALAH LAMOST Deacon Casagrande".split()
            for i,cat in enumerate(catalogs):
                _ = plot_Toomre_diagram(d, color=color, source=cat, ax=ax[i])
        fig.suptitle(self.cluster_name)
        return fig

    def plot_Toomre_diagram(self, color="Fe_H", source="Casagrande", ax=None):
        """Toomre diagram

        color : str
            stellar parameter used in color-coding scatter plot (default=Fe_H)
        source : str
            APOGEE, RAVE, GALAH, LAMOST, Deacon, & Casagrande
        """
        d = self.query_members_in_TGv8_catalog()
        column = "_".join([source,color])
        if len(d[column].dropna())==0:
            print(f"No data for {column}")
        else:
            u=d[f"Marchetti_U"]
            v=d[f"Marchetti_V"]
            w=d[f"Marchetti_W"]
            h=np.hypot(u,w)

            if ax is None:
                fig, ax = pl.subplots(figsize=(6,6))
            ax.set_title(source)
            cbar = ax.scatter(v, h, c=d[column])
            ax.set_xlabel("V [km/s]")
            ax.set_ylabel(r"$\sqrt{U^2+W^2}$ [km/s]")
            label = "_".join(color)
            pl.colorbar(cbar, ax=ax, label=label)
            return ax

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

        FIXME:
        If cluster size<5 arcmin:
            query a large radius and use gaia_sources.source_id.isin([])
        """
        # fp=join(data_loc,f'TablesGaiaDR2HRDpaper/{cluster_name}_members.hdf5')
        if self.cluster_members is None:
            df = self.query_cluster_members()
        if df is None:
            df = self.cluster_members

        fp = Path(data_loc, f"{self.cluster_name}_members.hdf5")
        if not fp.exists() or clobber:
            gaia_data = {}
            if top_n_brighest is not None:
                # sort in decreasing magnitude
                df = df.sort_values("phot_g_mean_mag", ascending=True)
                # use only top_n_brighest
                gaiaids = df.iloc[:top_n_brighest]["source_id"].values
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
                    t = target.Target(
                        gaiaDR2id=gaiaid,
                        verbose=self.verbose,
                        mission=self.mission,
                    )
                    df_gaia = t.query_gaia_dr2_catalog(
                        radius=radius, return_nearest_xmatch=True
                    )
                    gaia_data[gaiaid] = df_gaia
                except Exception as e:
                    print(e)
            # save
            errmsg = "data is empty"
            assert len(gaia_data) > 0, errmsg
            dd.io.save(fp, gaia_data)
            msg = f"Saved: {fp}"
        else:
            gaia_data = dd.io.load(fp)
            msg = f"Loaded: {fp}"
            errmsg = "data is empty"
            assert len(gaia_data) > 0, errmsg
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
        df_gaia.source_id = df_gaia.source_id.astype(int)
        self.cluster_members_gaia_params = df_gaia
        return df_gaia

    def plot_cluster_members(
        self,
        x=None,
        y=None,
        z=None,
        c=None,
        sigma=None,
        cmap=pl.get_cmap("Blues"),
    ):
        """visualize scatter plot and parallax density"""
        if self.cluster_members is None:
            df = self.query_cluster_members()
        else:
            df = self.cluster_members
        _print_warning(df)

        if (x is None) and (y is None):
            print(f"Choose `x`, `y`, `z` from\n{df.columns.tolist()}")
        else:
            fig, ax = pl.subplots(
                1, 2, figsize=(12, 5), constrained_layout=True
            )
            _ = df.plot(ax=ax[0], kind="scatter", x=x, y=y, c=c, cmap=cmap)
            # if c is not None:
            #     fig.colorbar(cbar, ax=ax, label=c)
            if sigma is not None:
                ax[0].plot(df[x].mean(), df[y].mean(), "ro", ms=10)
                ell = Ellipse(
                    (df[x].mean(), df[y].mean()),
                    width=df[x].std() * sigma,
                    height=df[y].std() * sigma,
                    color="r",
                    fill=False,
                )
                ax[0].add_artist(ell)
            ax = ax[1]
            _ = df[z].plot(ax=ax, kind="kde")
            ax.set_xlabel(z)
            fig.suptitle(self.cluster_name)
            return fig

    def plot_cmd(self, **kwargs):
        if self.cluster_members_gaia_params is None:
            gaia_params = self.query_cluster_members_gaia_params()
        else:
            gaia_params = self.cluster_members_gaia_params
        idx = gaia_params.source_id.astype(int).isin(
            self.cluster_members.source_id
        )
        gaia_params = gaia_params[idx]

        ax = plot_cmd(df=gaia_params, **kwargs)
        ax.set_title(self.cluster_name)
        return ax

    def plot_hrd(self, **kwargs):
        if self.cluster_members_gaia_params is None:
            gaia_params = self.query_cluster_members_gaia_params()
        else:
            gaia_params = self.cluster_members_gaia_params
        idx = gaia_params.source_id.astype(int).isin(
            self.cluster_members.source_id
        )
        gaia_params = gaia_params[idx]

        ax = plot_hrd(df=gaia_params, **kwargs)
        ax.set_title(self.cluster_name)
        return ax

    def plot_xyz_3d(self, **kwargs):
        if self.cluster_members_gaia_params is None:
            gaia_params = self.query_cluster_members_gaia_params()
        else:
            gaia_params = self.cluster_members_gaia_params
        idx = gaia_params.source_id.astype(int).isin(
            self.cluster_members.source_id
        )
        gaia_params = gaia_params[idx]

        fig = plot_xyz_3d(df=gaia_params, **kwargs)
        fig.suptitle(self.cluster_name)
        return fig

    def plot_rdp_pmrv(self, **kwargs):
        if self.cluster_members_gaia_params is None:
            gaia_params = self.query_cluster_members_gaia_params()
        else:
            gaia_params = self.cluster_members_gaia_params
        idx = gaia_params.source_id.astype(int).isin(
            self.cluster_members.source_id
        )
        gaia_params = gaia_params[idx]

        fig = plot_rdp_pmrv(df=gaia_params, **kwargs)
        fig.suptitle(self.cluster_name)
        return fig

    def plot_xyz_uvw(self, **kwargs):
        if self.cluster_members_gaia_params is None:
            gaia_params = self.query_cluster_members_gaia_params()
        else:
            gaia_params = self.cluster_members_gaia_params
        idx = gaia_params.source_id.astype(int).isin(
            self.cluster_members.source_id
        )
        gaia_params = gaia_params[idx]

        fig = plot_xyz_uvw(df=gaia_params, **kwargs)
        fig.suptitle(self.cluster_name)
        return fig

    def __call__(self):
        pass


def plot_cmd(
    df,
    target_gaiaid=None,
    match_id=True,
    df_target=None,
    target_label=None,
    log_age=None,
    feh=0.0,
    eep_limits=(202, 454),
    target_color="r",
    xaxis="bp_rp0",
    yaxis="abs_gmag",
    color="radius_val",
    figsize=(8, 8),
    estimate_color=False,
    cmap="viridis",
    ax=None,
):
    """Plot color-magnitude diagram using absolute G magnitude and dereddened Bp-Rp from Gaia photometry

    Parameters
    ----------
    df : pd.DataFrame
        cluster member properties
    match_id : bool
        checks if target gaiaid in df
    df_target : pd.Series
        info of target
    estimate_color : bool
        estimate absolute/dereddened color from estimated excess
    log_age : float
        isochrone age (default=None)
    feh : float
        isochrone metallicity
    eep_limits : tuple
        maximum eep (default=(202,454): (ZAMS,TAMS))

    Returns
    -------
    ax : axis
    """
    assert len(df) > 0, "df is empty"
    df["parallax"] = df["parallax"].astype(float)
    idx = ~np.isnan(df["parallax"]) & (df["parallax"] > 0)
    df = df[idx]
    if sum(~idx) > 0:
        print(f"{sum(~idx)} removed NaN or negative parallaxes")
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=figsize, constrained_layout=True)

    df["distance"] = Distance(parallax=df["parallax"].values * u.mas).pc
    # compute absolute Gmag
    df["abs_gmag"] = get_absolute_gmag(
        df["phot_g_mean_mag"], df["distance"], df["a_g_val"]
    )
    # compute intrinsic color index
    if estimate_color:
        df["bp_rp0"] = get_absolute_color_index(
            df["a_g_val"], df["phot_bp_mean_mag"], df["phot_rp_mean_mag"]
        )
    else:
        df["bp_rp0"] = df["bp_rp"] - df["e_bp_min_rp_val"]

    if target_gaiaid is not None:
        idx = df.source_id.astype(int).isin([target_gaiaid])
        if match_id:
            errmsg = f"Given cluster catalog does not contain the target gaia id [{target_gaiaid}]"
            assert sum(idx) > 0, errmsg
            x, y = df.loc[idx, "bp_rp0"], df.loc[idx, "abs_gmag"]
        else:
            assert df_target is not None, "provide df_target"
            df_target["distance"] = Distance(
                parallax=df_target["parallax"] * u.mas
            ).pc
            # compute absolute Gmag
            df_target["abs_gmag"] = get_absolute_gmag(
                df_target["phot_g_mean_mag"],
                df_target["distance"],
                df_target["a_g_val"],
            )
            # compute intrinsic color index
            if estimate_color:
                df_target["bp_rp0"] = get_absolute_color_index(
                    df_target["a_g_val"],
                    df_target["phot_bp_mean_mag"],
                    df_target["phot_rp_mean_mag"],
                )
            else:
                df_target["bp_rp0"] = (
                    df_target["bp_rp"] - df_target["e_bp_min_rp_val"]
                )
            x, y = df_target["bp_rp0"], df_target["abs_gmag"]
        if target_label is not None:
            ax.legend(loc="best")
        ax.plot(
            x,
            y,
            marker=r"$\star$",
            c=target_color,
            ms="25",
            label=target_label,
        )
    if log_age is not None:
        # plot isochrones
        try:
            from isochrones import get_ichrone

            iso_grid = get_ichrone("mist")
        except Exception:
            errmsg = "pip install isochrones"
        assert len(eep_limits) == 2, "eep_limits=(min,max)"
        iso_df = iso_grid.isochrone(log_age, feh)
        idx = (iso_df.eep > eep_limits[0]) & (iso_df.eep < eep_limits[1])
        G = iso_df.G_mag[idx]
        BP_RP = iso_df.BP_mag[idx] - iso_df.RP_mag[idx]
        label = f"log(t)={log_age:.2f}\nfeh={feh:.2f}"
        ax.plot(BP_RP, G, c="k", label=label)
        ax.legend(title="MIST isochrones")

    # df.plot.scatter(ax=ax, x="bp_rp", y="abs_gmag", marker=".")
    if color == "radius_val":
        rstar = np.log10(df[color].astype(float))
        c = ax.scatter(df[xaxis], df[yaxis], marker=".", c=rstar, cmap=cmap)
        fig.colorbar(c, ax=ax, label=r"$\log$(R/R$_{\odot}$)")
    else:
        ax.scatter(df[xaxis], df[yaxis], marker=".")
    ax.set_xlabel(r"$G_{BP} - G_{RP}$ [mag]", fontsize=16)
    ax.set_xlim(df[xaxis].min(), df[xaxis].max())
    ax.invert_yaxis()
    ax.set_ylabel(r"$G$ [mag]", fontsize=16)

    text = len(df[["bp_rp0", "abs_gmag"]].dropna())
    ax.text(0.8, 0.8, f"n={text}", fontsize=14, transform=ax.transAxes)
    return ax


def plot_hrd(
    df,
    target_gaiaid=None,
    match_id=True,
    df_target=None,
    target_label=None,
    target_color="r",
    log_age=None,
    feh=0.0,
    eep_limits=(202, 454),
    figsize=(8, 8),
    yaxis="lum_val",
    xaxis="teff_val",
    color="radius_val",
    cmap="viridis",
    annotate_Sun=False,
    ax=None,
):
    """Plot HR diagram using luminosity and Teff
    and optionally MIST isochrones if log_age is given

    Parameters
    ----------
    df : pd.DataFrame
        cluster memeber properties
    match_id : bool
        checks if target gaiaid in df
    df_target : pd.Series
        info of target
    log_age : float
        isochrone age (default=None)
    feh : float
        isochrone metallicity
    eep_limits : tuple
        maximum eep (default=(202,454): (ZAMS,TAMS))
    xaxis, yaxis : str
        parameter to plot

    Returns
    -------
    ax : axis
    """
    assert len(df) > 0, "df is empty"
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=figsize, constrained_layout=True)
    if target_gaiaid is not None:
        idx = df.source_id.astype(int).isin([target_gaiaid])
        if match_id:
            errmsg = f"Given cluster catalog does not contain the target gaia id [{target_gaiaid}]"
            assert sum(idx) > 0, errmsg
            x, y = df.loc[idx, xaxis], df.loc[idx, yaxis]
        else:
            assert df_target is not None, "provide df_target"
            df_target["distance"] = Distance(
                parallax=df_target["parallax"] * u.mas
            ).pc
            x, y = df_target[xaxis], df_target[yaxis]
        if target_label is not None:
            ax.legend(loc="best")
        ax.plot(
            x,
            y,
            marker=r"$\star$",
            c=target_color,
            ms="25",
            label=target_label,
        )
    # df.plot.scatter(ax=ax, x="bp_rp", y="abs_gmag", marker=".")
    rstar = np.log10(df[color].astype(float))
    # luminosity can be computed from abs mag; note Mag_sun = 4.85
    # df["abs_gmag"] = get_absolute_gmag(
    #     df["phot_g_mean_mag"], df["distance"], df["a_g_val"])
    # df["lum_val"] = 10**(0.4*(4.85-df["abs_gmag"])
    if color == "radius_val":
        c = ax.scatter(df[xaxis], df[yaxis], marker=".", c=rstar, cmap=cmap)
        fig.colorbar(c, ax=ax, label=r"$\log$(R/R$_{\odot}$)")
    else:
        ax.scatter(df[xaxis], df[yaxis], marker=".")

    if annotate_Sun:
        assert (yaxis == "lum_val") & (xaxis == "teff_val")
        ax.plot(5700, 1, marker=r"$\odot$", c="r", ms="15", label="Sun")
    if log_age is not None:
        # plot isochrones
        try:
            from isochrones.mist import MISTIsochroneGrid

            iso_grid = MISTIsochroneGrid()
            # from isochrones import get_ichrone
            # iso_grid = get_ichrone('mist').model_grid
        except Exception:
            errmsg = "pip install isochrones"
            raise ModuleNotFoundError(errmsg)
        assert len(eep_limits) == 2, "eep_limits=(min,max)"
        # check log_age
        ages = iso_grid.df.index.get_level_values(0)
        nearest_log_age = min(ages, key=lambda x: abs(x - log_age))
        errmsg = f"log_age={log_age} not in:\n{[round(x,2) for x in ages.unique().tolist()]}"
        # assert ages.isin([log_age]).any(), errmsg
        assert abs(nearest_log_age - log_age) < 0.1, errmsg
        # check feh
        fehs = iso_grid.df.index.get_level_values(1)
        nearest_feh = min(fehs, key=lambda x: abs(x - feh))
        errmsg = f"feh={feh} not in:\n{[round(x,2) for x in fehs.unique().tolist()]}"
        # assert fehs.isin([feh]).any(), errmsg
        assert abs(nearest_feh - feh) < 0.1, errmsg
        # get isochrone
        iso_df = iso_grid.df.loc[nearest_log_age, nearest_feh]
        iso_df["L"] = iso_df["logL"].apply(lambda x: 10 ** x)
        iso_df["Teff"] = iso_df["logTeff"].apply(lambda x: 10 ** x)
        label = f"log(t)={log_age:.2f}\nfeh={feh:.2f}"
        # limit eep
        idx = (iso_df.eep > eep_limits[0]) & (iso_df.eep < eep_limits[1])
        iso_df[idx].plot(x="Teff", y="L", c="k", ax=ax, label=label)
        ax.set_xlim(df[xaxis].min() - 100, df[xaxis].max() + 100)
        ax.legend(title="MIST isochrones")

    ax.set_ylabel(r"$L/L_{\odot}$", fontsize=16)
    ax.invert_xaxis()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T_{\rm{eff}}$/K", fontsize=16)
    text = len(df[[xaxis, yaxis]].dropna())
    ax.text(0.8, 0.8, f"nstars={text}", fontsize=14, transform=ax.transAxes)
    return ax


def plot_rdp_pmrv(
    df,
    target_gaiaid=None,
    match_id=True,
    df_target=None,
    target_label=None,
    target_color="r",
    color="teff_val",
    marker="o",
    figsize=(10, 10),
    cmap="viridis",
):
    """
    Plot ICRS position and proper motions in 2D scatter plots,
    and parallax and radial velocity in kernel density

    Parameters
    ----------
    df : pandas.DataFrame
        contains ra, dec, parallax, pmra, pmdec, radial_velocity columns
    target_gaiaid : int
        target gaia DR2 id
    """
    assert len(df) > 0, "df is empty"
    fig, axs = pl.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax = axs.flatten()

    n = 1
    x, y = "ra", "dec"
    # _ = df.plot.scatter(x=x, y=y, c=color, marker=marker, ax=ax[n], cmap=cmap)
    c = df[color] if color is not None else None
    cbar = ax[n].scatter(df[x], df[y], c=c, marker=marker, cmap=cmap)
    if color is not None:
        fig.colorbar(cbar, ax=ax[n], label=color)
    if target_gaiaid is not None:
        idx = df.source_id.astype(int).isin([target_gaiaid])
        if match_id:
            errmsg = f"Given cluster does not contain the target gaia id [{target_gaiaid}]"
            assert sum(idx) > 0, errmsg
            ax[n].plot(
                df.loc[idx, x],
                df.loc[idx, y],
                marker=r"$\star$",
                c=target_color,
                ms="25",
                label=target_label,
            )
        else:
            assert df_target is not None, "provide df_target"
            ax[n].plot(
                df_target[x],
                df_target[y],
                marker=r"$\star$",
                c=target_color,
                ms="25",
                label=target_label,
            )
    ax[n].set_xlabel("R.A. [deg]")
    ax[n].set_ylabel("Dec. [deg]")
    text = len(df[["ra", "dec"]].dropna())
    ax[n].text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes)
    if target_label is not None:
        ax[n].legend(loc="best")
    n = 0
    par = "parallax"
    df[par].plot.kde(ax=ax[n])
    if target_gaiaid is not None:
        idx = df.source_id.astype(int).isin([target_gaiaid])
        if match_id:
            errmsg = f"Given cluster does not contain the target gaia id [{target_gaiaid}]"
            assert sum(idx) > 0, errmsg
            ax[n].axvline(
                df.loc[idx, par].values[0],
                0,
                1,
                c="k",
                ls="--",
                label=target_label,
            )
        else:
            assert df_target is not None, "provide df_target"
            ax[n].axvline(
                df_target[par], 0, 1, c="k", ls="--", label=target_label
            )

        if target_label is not None:
            ax[n].legend(loc="best")
    ax[n].set_xlabel("Parallax [mas]")
    text = len(df[par].dropna())
    ax[n].text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes)
    n = 3
    x, y = "pmra", "pmdec"
    # _ = df.plot.scatter(x=x, y=y, c=c, marker=marker, ax=ax[n], cmap=cmap)
    c = df[color] if color is not None else None
    cbar = ax[n].scatter(df[x], df[y], c=c, marker=marker, cmap=cmap)
    if (color is not None) & (n == 3):
        # show last colorbar only
        fig.colorbar(cbar, ax=ax[n], label=color)
    if target_gaiaid is not None:
        idx = df.source_id.astype(int).isin([target_gaiaid])
        if match_id:
            errmsg = f"Given cluster does not contain the target gaia id [{target_gaiaid}]"
            assert sum(idx) > 0, errmsg
            ax[n].plot(
                df.loc[idx, x],
                df.loc[idx, y],
                marker=r"$\star$",
                c=target_color,
                ms="25",
            )
        else:
            assert df_target is not None, "provide df_target"
            ax[n].plot(
                df_target[x],
                df_target[y],
                marker=r"$\star$",
                c=target_color,
                ms="25",
            )
    ax[n].set_xlabel("PM R.A. [deg]")
    ax[n].set_ylabel("PM Dec. [deg]")
    text = len(df[["pmra", "pmdec"]].dropna())
    ax[n].text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes)
    n = 2
    par = "radial_velocity"
    try:
        df[par].plot.kde(ax=ax[n])
        if target_gaiaid is not None:
            idx = df.source_id.astype(int).isin([target_gaiaid])
            if match_id:
                errmsg = f"Given cluster does not contain the target gaia id [{target_gaiaid}]"
                assert sum(idx) > 0, errmsg
                ax[n].axvline(
                    df.loc[idx, par].values[0],
                    0,
                    1,
                    c="k",
                    ls="--",
                    label=target_label,
                )
            else:
                ax[n].axvline(
                    df_target[par], 0, 1, c="k", ls="--", label=target_label
                )
        ax[n].set_xlabel("RV [km/s]")
        text = len(df[par].dropna())
        ax[n].text(
            0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes
        )
    except Exception as e:
        print(e)
        # catalog_name = df.Cluster.unique()()
        raise ValueError(
            f"radial_velocity is not available"
        )  # in {catalog_name}
    return fig


def plot_xyz_uvw(
    df,
    target_gaiaid=None,
    match_id=True,
    df_target=None,
    target_color="r",
    color="teff_val",
    marker="o",
    verbose=True,
    figsize=(12, 8),
    cmap="viridis",
):
    """
    Plot 3D position in galactocentric (xyz) frame
    and proper motion with radial velocity in galactic cartesian velocities
    (UVW) frame

    Parameters
    ----------
    df : pandas.DataFrame
        contains ra, dec, parallax, pmra, pmdec, radial_velocity columns
    target_gaiaid : int
        target gaia DR2 id
    df_target : pandas.Series
        target's gaia parameters

    Note: U is positive towards the direction of the Galactic center (GC);
    V is positive for a star with the same rotational direction as the Sun going around the galaxy,
    with 0 at the same rotation as sources at the Sun’s distance,
    and W positive towards the north Galactic pole

    U,V,W can be converted to Local Standard of Rest (LSR) by subtracting V = 238 km/s,
    the adopted rotation velocity at the position of the Sun from Marchetti et al. (2018).

    See also https://arxiv.org/pdf/1707.00697.pdf which estimates Sun's
    (U,V,W) = (9.03, 255.26, 7.001)

    See also https://arxiv.org/pdf/1804.10607.pdf for modeling Gaia DR2 in 6D
    """
    assert len(df) > 0, "df is empty"
    fig, axs = pl.subplots(2, 3, figsize=figsize, constrained_layout=True)
    ax = axs.flatten()

    if not np.all(df.columns.isin("X Y Z U V W".split())):
        df = get_transformed_coord(df, frame="galactocentric", verbose=verbose)
    if df_target is not None:
        df_target = get_transformed_coord(
            pd.DataFrame(df_target).T, frame="galactocentric"
        )
    n = 0
    for (i, j) in itertools.combinations(["X", "Y", "Z"], r=2):
        if target_gaiaid is not None:
            idx = df.source_id.astype(int).isin([target_gaiaid])
            if match_id:
                errmsg = f"Given cluster does not contain the target gaia id [{target_gaiaid}]"
                assert sum(idx) > 0, errmsg
                ax[n].plot(
                    df.loc[idx, i],
                    df.loc[idx, j],
                    marker=r"$\star$",
                    c=target_color,
                    ms="25",
                )
            else:
                assert df_target is not None, "provide df_target"
                ax[n].plot(
                    df_target[i],
                    df_target[j],
                    marker=r"$\star$",
                    c=target_color,
                    ms="25",
                )
        # _ = df.plot.scatter(x=i, y=j, c=color, marker=marker, ax=ax[n])
        c = df[color] if color is not None else None
        cbar = ax[n].scatter(df[i], df[j], c=c, marker=marker, cmap=cmap)
        # if color is not None:
        #     fig.colorbar(cbar, ax=ax[n], label=color)
        ax[n].set_xlabel(i + " [pc]")
        ax[n].set_ylabel(j + " [pc]")
        text = len(df[[i, j]].dropna())
        ax[n].text(
            0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes
        )
        n += 1

    n = 3
    for (i, j) in itertools.combinations(["U", "V", "W"], r=2):
        if target_gaiaid is not None:
            idx = df.source_id.astype(int).isin([target_gaiaid])
            if match_id:
                errmsg = f"Given cluster does not contain the target gaia id [{target_gaiaid}]"
                assert sum(idx) > 0, errmsg
                ax[n].plot(
                    df.loc[idx, i],
                    df.loc[idx, j],
                    marker=r"$\star$",
                    c=target_color,
                    ms="25",
                )
            else:
                ax[n].plot(
                    df_target[i],
                    df_target[j],
                    marker=r"$\star$",
                    c=target_color,
                    ms="25",
                )
        # _ = df.plot.scatter(x=i, y=j, c=color, marker=marker, ax=ax[n], cmap=cmap)
        c = df[color] if color is not None else None
        cbar = ax[n].scatter(df[i], df[j], c=c, marker=marker, cmap=cmap)
        if (color is not None) and (n == 5):
            # show last colorbar only only
            fig.colorbar(cbar, ax=ax[n], label=color)
        ax[n].set_xlabel(i + " [km/s]")
        ax[n].set_ylabel(j + " [km/s]")
        text = len(df[[i, j]].dropna())
        ax[n].text(
            0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes
        )
        n += 1

    return fig


def plot_xyz_3d(
    df,
    target_gaiaid=None,
    match_id=True,
    df_target=None,
    target_color="r",
    color="teff_val",
    marker="o",
    xlim=None,
    ylim=None,
    zlim=None,
    figsize=(8, 5),
    cmap="viridis",
):
    """plot 3-d position in galactocentric frame

    Parameters
    ----------
    df : pandas.DataFrame
        contains ra, dec & parallax columns
    target_gaiaid : int
        target gaia DR2 id
    xlim,ylim,zlim : tuple
        lower and upper bounds
    """
    fig = pl.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(30, 120)

    if "distance" not in df.columns:
        df["distance"] = Distance(parallax=df.parallax.values * u.mas).pc

    coords = SkyCoord(
        ra=df.ra.values * u.deg,
        dec=df.dec.values * u.deg,
        distance=df.distance.values * u.pc,
    )
    xyz = coords.galactocentric
    df["x"] = xyz.x
    df["y"] = xyz.y
    df["z"] = xyz.z

    idx1 = np.zeros_like(df.x, dtype=bool)
    if xlim:
        assert isinstance(xlim, tuple)
        idx1 = (df.x > xlim[0]) & (df.x < xlim[1])
    idx2 = np.zeros_like(df.y, dtype=bool)
    if ylim:
        assert isinstance(ylim, tuple)
        idx2 = (df.y > ylim[0]) & (df.y < ylim[1])
    idx3 = np.zeros_like(df.z, dtype=bool)
    if zlim:
        assert isinstance(zlim, tuple)
        idx3 = (df.z > zlim[0]) & (df.z < zlim[1])
    idx = idx1 | idx2 | idx3
    c = df.loc[idx, color]
    cbar = ax.scatter(
        xs=df[idx].x,
        ys=df[idx].y,
        zs=df[idx].z,
        c=c,
        marker=marker,
        cmap=cmap,
        alpha=0.5,
    )
    if color is not None:
        fig.colorbar(cbar, ax=ax, label=color)
    idx = df.source_id == target_gaiaid
    ax.scatter(
        xs=df[idx].x,
        ys=df[idx].y,
        zs=df[idx].z,
        marker=r"$\star$",
        c=target_color,
        s=300,
    )
    pl.setp(ax, xlabel="X", ylabel="Y", zlabel="Z")
    return fig


def _print_warning(df, default_rows=50):
    """warn about partial table"""
    warning = f"WARNING: Table contains only the first {default_rows} rows.\n"
    warning += "Try `get_vizier_tables(row_limit=-1)` to download full table."
    if df.shape[0] == default_rows:
        print(warning)
    else:
        pass


def _decode_n_drop(df, columns):
    """
    columns : list of columns to drop
    """
    # bytes to str
    df = df.applymap(
        lambda x: x.decode("ascii") if isinstance(x, bytes) else x
    )
    # remove columns
    df = df.drop(columns, axis=1)
    return df

# -*- coding: utf-8 -*-

r"""
Module for general helper functions
"""

# Import standard library
import os
import logging
import itertools
import urllib
from pathlib import Path
from glob import glob
from operator import concat
from functools import reduce
from os.path import join, exists
from pprint import pprint

# Import from module
# from matplotlib.figure import Figure
# from matplotlib.image import AxesImage
# from loguru import logger
from uncertainties import unumpy
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.ndimage import zoom
import matplotlib.pyplot as pl
import lightkurve as lk
from astropy.visualization import hist
from astropy import units as u
from astropy import constants as c
from astropy.timeseries import LombScargle
from astropy.modeling import models, fitting
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import (
    SkyCoord,
    Distance,
    sky_coordinate,
    Galactocentric,
    match_coordinates_3d,
)
from skimage import measure
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs, Tesscut
from astroquery.gaia import Gaia
import deepdish as dd

# Import from package
from chronos import target
from chronos import cluster
from chronos import gls
from chronos.config import DATA_PATH

log = logging.getLogger(__name__)

__all__ = [
    "get_k2_data_from_exofop",
    "get_nexsci_archive",
    "get_tepcat",
    "get_tess_ccd_info",
    "get_all_campaigns",
    "get_all_sectors",
    "get_sector_cam_ccd",
    "get_tois",
    "get_toi",
    "get_ctois",
    "get_ctoi",
    "get_target_coord",
    "get_epicid_from_k2name",
    "get_target_coord_3d",
    "get_transformed_coord",
    "query_gaia_params_of_all_tois",
    "get_mamajek_table",
    "get_distance",
    "get_excess_from_extiction",
    "get_absolute_color_index",
    "get_absolute_gmag",
    "parse_aperture_mask",
    "make_round_mask",
    "make_square_mask",
    "remove_bad_data",
    "is_point_inside_mask",
    "get_fluxes_within_mask",
    "get_harps_bank",
    "get_specs_table_from_tfop",
    "get_rotation_period",
    "get_transit_mask",
    "get_mag_err_from_flux",
    "get_phase",
    "bin_data",
    "map_float",
    "map_int",
    "flatten_list",
    "detrend",
    "query_tpf",
    "query_tpf_tesscut",
    "is_gaiaid_in_cluster",
    "get_pix_area_threshold",
    "get_above_lower_limit",
    "get_below_upper_limit",
    "get_between_limits",
    "get_RV_K",
    "get_RM_K",
    "get_tois_mass_RV_K",
    "get_vizier_tables",
    "get_mist_eep_table",
    "get_max_dmag_from_depth",
    "get_TGv8_catalog",
    "get_tois_in_TGv8_catalog",
    "get_filter_transmission_from_SVO",
    "get_limbdark",
    "get_secondary_eclipse_threshold",
]

# Ax/Av
extinction_ratios = {
    "U": 1.531,
    "B": 1.324,
    "V": 1.0,
    "R": 0.748,
    "I": 0.482,
    "J": 0.282,
    "H": 0.175,
    "K": 0.112,
    "G": 0.85926,
    "Bp": 1.06794,
    "Rp": 0.65199,
}


def query_asas_sn_catalog():
    """
    NASA/IRSA (has SED viewer also):
    https://irsa.ipac.caltech.edu/frontpage/
    ASAS:
    https://asas-sn.osu.edu/photometry
    http://www.astrouw.edu.pl/asas/?page=acvs
    """
    raise NotImplementedError()


def get_k2_data_from_exofop(epic, table="star"):
    """
    get data from exofop table
    """
    keys = {
        "phot": 1,
        "mag": 1,
        "star": 2,
        "planet": 3,
        "spec": 4,
        "imaging": 5,
        "file": 8,
    }
    errmsg = f"table={table} not in\n{list(keys.keys())}"
    assert table in list(keys.keys()), errmsg
    key = keys[table]
    url = f"https://exofop.ipac.caltech.edu/k2/edit_target.php?id={epic}"
    data = pd.read_html(url, attrs={"id": f"myTable{key}"})[0]
    # remove multi-index column
    data = data.T.reset_index(level=0, drop=True).T
    data["epic"] = epic
    return data


def get_filter_transmission_from_SVO(
    filter,
    telescope,
    instrument=None,
    plot=False,
    format="ascii",
    verbose=True,
):
    """
    get filter response functions from http://svo2.cab.inta-csic.es/theory/fps/

    Parameters
    ----------
    filter : str
        filter
    telescope : str
        observatory name
    instrument : str
        instrument (optional)

    Returns
    -------
    df : pandas.DataFrame
    """
    try:
        from bs4 import BeautifulSoup
    except Exception:
        cmd = "pip install beautifulsoup4"
        raise ModuleNotFoundError(cmd)

    base_url = "http://svo2.cab.inta-csic.es/theory/fps/"

    # get urls for each telescope/facility
    with urllib.request.urlopen(base_url) as response:
        html = response.read()
    soup = BeautifulSoup(html)
    html_urls = soup.findAll("a")

    urls = []
    telescopes = {}
    for html_url in html_urls:
        url = html_url.get("href")
        if "index.php?mode=browse&gname=" in url:
            filter_name = url.split("&gname=")[-1]
            urls.append(url)
            telescopes[filter_name] = url
    if telescope not in telescopes.keys():
        errmsg = f"{telescope} not in available telescopes:"
        errmsg += f"\n{list(telescopes.keys())}"
        raise ValueError(errmsg)

    filter_url = base_url + telescopes[telescope]
    if instrument is not None:
        filter_url += f"&gname2={instrument}"
    # get filters of chosen telescope
    if verbose:
        print(f"Querying {filter_url}")
    with urllib.request.urlopen(filter_url) as response:
        html = response.read()
    soup = BeautifulSoup(html)
    html_urls = soup.findAll("a")

    instrument_filters = {}
    instruments = []
    for html_url in html_urls:
        url = html_url.get("href")
        if "#filter" in url:
            instrument_filter = url.split(f"{telescope}/")[1].split("&&mode=")[
                0
            ]
            instrument_name = instrument_filter.split(".")[0]
            filter_name = instrument_filter.split(".")[1]
            instrument_filters[filter_name] = instrument_filter
            instruments.append(instrument_name)

    if instrument is None:
        instrument = np.unique(instruments)[0]

    errmsg = f"{instrument} not available.\nSee {filter_url}"
    assert np.any(instrument in instruments), errmsg

    if filter not in instrument_filters.keys():
        errmsg = f"{filter} not in available {instrument} filters:"
        errmsg += f"\n{list(instrument_filters.keys())}"
        raise ValueError(errmsg)

    dl_url = f"getdata.php?format={format}&id={telescope}/{instrument_filters[filter]}"
    try:
        full_url = base_url + dl_url
        df = pd.read_csv(
            full_url, names=["wav_nm", "transmission"], delim_whitespace=True
        )
        df = df[df.transmission > 0]
        df["wav_nm"] = df.wav_nm / 10

        if plot:
            ax = df.plot(
                x="wav_nm", y="transmission", label=f"{telescope}/{filter}"
            )
            ax.set_xlabel("wavelength [nm]")
            ax.set_ylabel("Transmission")
        return df
    except Exception as e:
        raise (f"Error: {e}\nCheck url: {full_url}")


def get_tois_in_TGv8_catalog(query_str=None, data_path=None):
    """
    Thin, Thick, Halo disc classification of cross-matched TOIs
    See Carillo+2019: https://arxiv.org/pdf/1911.07825.pdf
    Returned df can then be plotted: `plot_Toomre(df)`

    query_str : str
        e.g. TD>0.9 (Thick disk prob > 90%)
    """
    df = get_TGv8_catalog(data_path=data_path)
    # pre-querried gaia params
    toi_params = query_gaia_params_of_all_tois(update=False)
    # match
    idx = df.Gaia_source_id.isin(toi_params.source_id)
    if query_str is None:
        return df[idx]
    else:
        df2 = df.query(query_str)
        # match query with toi
        toi_params2 = toi_params[toi_params.source_id.isin(df2.Gaia_source_id)]
        toi_params2 = toi_params2.rename(
            {"source_id": "Gaia_source_id"}, axis=1
        )
        return toi_params2.reset_index().merge(df2, on="Gaia_source_id")


def get_TGv8_catalog(data_path=None):
    """
    Stellar parameters of TESS host stars (TICv8)
    using Gaia2+APOGEE14+GALAH+RAVE5+LAMOST+SkyMapper;
    See Carillo2020: https://arxiv.org/abs/1911.07825
    Parameter
    ---------
    data_path : str
        path to data

    Returns
    -------
    pandas.DataFrame
    """
    zenodo_url = "https://zenodo.org/record/3546184#.Xt-UFIFq1Ol"
    if data_path is None:
        fp = Path(DATA_PATH, "TGv8sample_vs_surveys.fits")
    else:
        fp = Path(data_path, "TGv8sample_vs_surveys.fits")

    if not Path(fp).exists():
        errmsg = f"Data is not found in {DATA_PATH}\n"
        errmsg += f"Download it first from {zenodo_url} (size~1Gb)"
        raise FileNotFoundError(errmsg)
    # with fits.open(fp) as hdulist:
    #     # print(hdulist.info())
    #     data = hdulist[1].data
    df = Table.read(fp).to_pandas()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def get_max_dmag_from_depth(depth):
    """maximum delta magnitude from transit depth"""
    return 2.5 * np.log10(depth)


def query_WDSC():
    """
    Washington Double Star Catalog
    """
    url = "http://www.astro.gsu.edu/wds/Webtextfiles/wdsnewframe.html"
    df = pd.read_csv(url)
    return df


def get_tepcat(catalog="all"):
    """
    TEPCat
    https://www.astro.keele.ac.uk/jkt/tepcat/

    Choices:
    all, homogenerous, planning, obliquity
    """
    base_url = "https://www.astro.keele.ac.uk/jkt/tepcat/"
    if catalog == "all":
        full_url = base_url + "allplanets-csv.csv"
    elif catalog == "homogeneous":
        full_url = base_url + "homogeneous-par-csv.csv"
    elif catalog == "planning":
        full_url = base_url + "observables.csv"
    elif catalog == "obliquity":
        full_url = base_url + "obliquity.csv"
    else:
        raise ValueError("catalog=[all,homogeneous,planning,obliquity]")
    df = pd.read_csv(full_url)
    return df


def get_mist_eep_table():
    """
    For eep phases, see
    http://waps.cfa.harvard.edu/MIST/README_tables.pdf
    """
    fp = Path(DATA_PATH, "mist_eep_table.csv")
    return pd.read_csv(fp, comment="#")


def get_nexsci_archive(table="all"):
    """
    direct download from NExSci archive
    """
    base_url = "https://exoplanetarchive.ipac.caltech.edu/"
    settings = "cgi-bin/nstedAPI/nph-nstedAPI?table="
    if table == "all":
        url = base_url + settings + "exomultpars"
    elif table == "confirmed":
        url = base_url + settings + "exoplanets"
    elif table == "composite":
        url = base_url + settings + "compositepars"
    else:
        raise ValueError("table=[all, confirmed, composite]")
    df = pd.read_csv(url)
    return df


def get_nexsci_candidates(cache=False):
    """
    """
    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
    except Exception:
        raise ModuleNotFoundError("pip install astroquery --update")
    candidates = NasaExoplanetArchive.query_criteria(
        table="k2candidates", cache=cache
    )
    nexsci_pc = candidates.to_pandas()
    # nexsci_pc = nexsci_pc.query("k2c_disp=='CONFIRMED'")
    return nexsci_pc.query("k2c_disp=='CANDIDATE'")


def get_vizier_tables(key, tab_index=None, row_limit=50, verbose=True):
    """
    Parameters
    ----------
    key : str
        vizier catalog key
    tab_index : int
        table index to download and parse
    Returns
    -------
    tables if tab_index is None else parsed df
    """
    if row_limit == -1:
        msg = "Downloading all tables in "
    else:
        msg = f"Downloading the first {row_limit} rows of each table in "
    msg += f"{key} from vizier."
    if verbose:
        print(msg)
    # set row limit
    Vizier.ROW_LIMIT = row_limit

    tables = Vizier.get_catalogs(key)
    errmsg = "No data returned from Vizier."
    assert tables is not None, errmsg

    if tab_index is None:
        if verbose:
            print({k: tables[k]._meta["description"] for k in tables.keys()})
        return tables
    else:
        df = tables[tab_index].to_pandas()
        df = df.applymap(
            lambda x: x.decode("ascii") if isinstance(x, bytes) else x
        )
        return df


def get_tois_mass_RV_K(clobber=False):
    fp = Path(DATA_PATH, "TOIs2.csv")
    if clobber:
        try:
            from mrexo import predict_from_measurement, generate_lookup_table
        except Exception:
            raise ModuleNotFoundError("pip install mrexo")

        tois = get_tois()

        masses = {}
        for key, row in tqdm(tois.iterrows()):
            toi = row["TOI"]
            Rp = row["Planet Radius (R_Earth)"]
            Rp_err = row["Planet Radius (R_Earth) err"]
            Mp, (Mp_lo, Mp_hi), iron_planet = predict_from_measurement(
                measurement=Rp,
                measurement_sigma=Rp_err,
                qtl=[0.16, 0.84],
                dataset="kepler",
            )
            masses[toi] = (Mp, Mp_lo, Mp_hi)

        df = pd.DataFrame(masses).T
        df.columns = [
            "Planet mass (Mp_Earth)",
            "Planet mass (Mp_Earth) lo",
            "Planet mass (Mp_Earth) hi",
        ]
        df.index.name = "TOI"
        df = df.reset_index()

        df["RV_K_lo"] = get_RV_K(
            tois["Period (days)"],
            tois["Stellar Radius (R_Sun)"],  # should be Mstar
            df["Planet mass (Mp_Earth) lo"],
            with_unit=True,
        )

        df["RV_K_hi"] = get_RV_K(
            tois["Period (days)"],
            tois["Stellar Radius (R_Sun)"],  # should be Mstar
            df["Planet mass (Mp_Earth) hi"],
            with_unit=True,
        )

        joint = pd.merge(tois, df, on="TOI")
        joint.to_csv(fp, index=False)
        print(f"Saved: {fp}")
    else:
        joint = pd.read_csv(fp)
        print(f"Loaded: {fp}")
    return joint


def get_phase(time, period, epoch, offset=0.5):
    """phase offset -0.5,0.5
    """
    phase = (((((time - epoch) / period) + offset) % 1) / offset) - 1
    return phase


def bin_data(array, binsize, func=np.mean):
    """
    """
    a_b = []
    for i in range(0, array.shape[0], binsize):
        a_b.append(func(array[i : i + binsize], axis=0))

    return a_b


def get_tess_ccd_info(target_coord):
    """use search_targetpixelfile like get_all_sectors?"""
    ccd_info = Tesscut.get_sectors(target_coord)
    errmsg = "Target not found in any TESS sectors"
    assert len(ccd_info) > 0, errmsg
    return ccd_info.to_pandas()


def get_all_sectors(target_coord):
    """ """
    ccd_info = get_tess_ccd_info(target_coord)
    all_sectors = [int(i) for i in ccd_info["sector"].values]
    return np.array(all_sectors)


def get_all_campaigns(epicid):
    """ """
    res = lk.search_targetpixelfile(
        f"K2 {epicid}", campaign=None, mission="K2"
    )
    errmsg = "No data found"
    assert len(res) > 0, errmsg
    df = res.table.to_pandas()
    campaigns = df["observation"].apply(lambda x: x.split()[-1]).values
    return np.array([int(c) for c in campaigns])


def get_sector_cam_ccd(target_coord, sector=None):
    """get TESS sector, camera, and ccd numbers using Tesscut
    """
    df = get_tess_ccd_info(target_coord)
    all_sectors = [int(i) for i in df["sector"].values]
    if sector is not None:
        sector_idx = df["sector"][df["sector"].isin([sector])].index.tolist()
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


def is_gaiaid_in_cluster(
    gaiaid, cluster_name=None, catalog_name="Bouma2019", verbose=True
):
    """
    See scripts/check_target_in_cluster
    """
    # reduce the redundant names above
    gaiaid = int(gaiaid)
    if cluster_name is None:
        cc = cluster.ClusterCatalog(catalog_name=catalog_name, verbose=False)
        df_mem = cc.query_catalog(return_members=True)
    else:
        c = cluster.Cluster(
            catalog_name=catalog_name, cluster_name=cluster_name, verbose=False
        )
        df_mem = c.query_cluster_members()
    idx = df_mem.source_id.isin([gaiaid])
    if idx.sum() > 0:
        if verbose:
            if cluster_name is None:
                cluster_match = df_mem[idx].Cluster.values[0]
            else:
                # TODO: what if cluster_match != cluster_name?
                cluster_match = cluster_name
            print(
                f"Gaia DR2 {gaiaid} is IN {cluster_match} cluster based on {catalog_name} catalog!"
            )
        return True
    else:
        if verbose:
            print(f"Gaia DR2 {gaiaid} is NOT in {catalog_name} catalog!")
        return False


def query_tpf(
    query_str,
    sector=None,
    campaign=None,
    quality_bitmask="default",
    apply_data_quality_mask=False,
    mission="TESS",
    verbose=True,
):
    """
    """
    if verbose:
        print(f"Searching targetpixelfile for {query_str} using lightkurve")

    tpf = lk.search_targetpixelfile(
        query_str, mission=mission, sector=sector, campaign=campaign
    ).download()
    if apply_data_quality_mask:
        tpf = remove_bad_data(tpf, sector=sector, verbose=verbose)
    return tpf


def query_tpf_tesscut(
    query_str,
    sector=None,
    quality_bitmask="default",
    cutout_size=(15, 15),
    apply_data_quality_mask=False,
    verbose=True,
):
    """
        """
    if verbose:
        if isinstance(query_str, sky_coordinate.SkyCoord):
            query = f"ra,dec=({query_str.to_string()})"
        else:
            query = query_str
        print(f"Searching targetpixelfile for {query} using Tesscut")
    tpf = lk.search_tesscut(query_str, sector=sector).download(
        quality_bitmask=quality_bitmask, cutout_size=cutout_size
    )
    assert tpf is not None, "No results from Tesscut search."
    # remove zeros
    zero_mask = (tpf.flux_err == 0).all(axis=(1, 2))
    if zero_mask.sum() > 0:
        tpf = tpf[~zero_mask]
    if apply_data_quality_mask:
        tpf = remove_bad_data(tpf, sector=sector, verbose=verbose)
    return tpf


def detrend(self, polyorder=1, break_tolerance=10):
    """mainly to be added as method to lk.LightCurve
    """
    lc = self.copy()
    half = lc.time.shape[0] // 2
    if half % 2 == 0:
        # add 1 if even
        half += 1
    return lc.flatten(
        window_length=half,
        polyorder=polyorder,
        break_tolerance=break_tolerance,
    )


def get_rotation_period(
    time,
    flux,
    flux_err=None,
    min_per=0.5,
    max_per=None,
    method="ls",
    npoints=20,
    plot=True,
    verbose=True,
):
    """
    time, flux : array
        time and flux
    min_period, max_period : float
        minimum & maxmimum period (default=half baseline e.g. ~13 days)
    method : str
        ls = lomb-scargle; gls = generalized ls
    npoints : int
        datapoints around which to fit a Gaussian

    Note:
    1. Transits are assumed to be masked already
    2. The period and uncertainty were determined from the mean and the
    half-width at half-maximum of a Gaussian fit to the periodogram peak, respectively
    See also:
    https://arxiv.org/abs/1702.03885
    """
    baseline = int(time[-1] - time[0])
    max_per = max_per if max_per is not None else baseline / 2

    if method == "ls":
        if verbose:
            print("Using Lomb-Scargle method")
        ls = LombScargle(time, flux, dy=flux_err)
        frequencies, powers = ls.autopower(
            minimum_frequency=1.0 / max_per, maximum_frequency=1.0 / min_per
        )
        idx = np.argmax(powers)
        while npoints > idx:
            npoints -= 1

        best_freq = frequencies[idx]
        best_period = 1.0 / best_freq
        # specify which points to fit a gaussian
        x = (1 / frequencies)[idx - npoints : idx + npoints]
        y = powers[idx - npoints : idx + npoints]

        # Fit the data using a 1-D Gaussian
        g_init = models.Gaussian1D(amplitude=0.5, mean=best_period, stddev=1)
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, x, y)

        label = f"P={g.mean.value:.2f}+/-{g.stddev.value:.2f} d"
        if plot:
            # Plot the data with the best-fit model
            pl.plot(x, y, "ko", label="_nolegend_")
            pl.plot(x, g(x), label="_nolegend_")
            pl.ylabel("Lomb-Scargle Power")
            pl.xlabel("Period [days]")
            pl.axvline(g.mean, 0, 1, ls="--", c="r", label=label)
            pl.legend()

        if verbose:
            print(label)

        return (g.mean.value, g.stddev.value)

    elif method == "gls":
        if verbose:
            print("Using Generalized Lomb-Scargle method")
        data = (time, flux, flux_err)
        ls = gls.Gls(data, Pbeg=min_per, Pend=max_per, verbose=verbose)
        prot, prot_err = ls.hpstat["P"], ls.hpstat["e_P"]
        if plot:
            _ = ls.plot(block=False, figsize=(10, 8))
        return (prot, prot_err)
    else:
        raise ValueError("Use method=[ls | gls]")


def get_transit_mask(lc, period, epoch, duration_hours):
    """
    lc : lk.LightCurve
        lightcurve that contains time and flux properties

    mask = []
    t0 += np.ceil((time[0] - dur - t0) / period) * period
    for t in np.arange(t0, time[-1] + dur, period):
        mask.extend(np.where(np.abs(time - t) < dur / 2.)[0])
    return  np.array(mask)
    """
    assert isinstance(lc, lk.LightCurve)
    assert (
        (period is not None)
        & (epoch is not None)
        & (duration_hours is not None)
    )
    temp_fold = lc.fold(period, t0=epoch)
    fractional_duration = (duration_hours / 24.0) / period
    phase_mask = np.abs(temp_fold.phase) < (fractional_duration * 1.5)
    transit_mask = np.in1d(lc.time, temp_fold.time_original[phase_mask])
    return transit_mask


def get_harps_bank(
    target_coord, separation=30, outdir=DATA_PATH, verbose=True
):
    """
    Check if target has archival HARPS data from:
    http://www.mpia.de/homes/trifonov/HARPS_RVBank.html
    See also https://github.com/3fon3fonov/HARPS_RVBank

    For column meanings:
    https://www2.mpia-hd.mpg.de/homes/trifonov/HARPS_RVBank_header.txt
    """
    homeurl = "http://www.mpia.de/homes/trifonov/HARPS_RVBank.html"
    fp = os.path.join(outdir, "HARPS_RVBank_table.csv")
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        msg = f"Loaded: {fp}\n"
    else:
        if verbose:
            print(
                f"Downloading HARPS bank from {homeurl}. This may take a while."
            )
        # csvurl = "http://www.mpia.de/homes/trifonov/HARPS_RVBank_v1.csv"
        # df = pd.read_csv(csvurl)
        df = pd.read_html(homeurl, header=0)[0]  # choose first table
        df.to_csv(fp, index=False)
        msg = f"Saved: {fp}\n"
    if verbose:
        print(msg)
    # coordinates
    coords = SkyCoord(
        ra=df["RA"],
        dec=df["DEC"],
        distance=df["Dist [pc]"],
        unit=(u.hourangle, u.deg, u.pc),
    )
    # check which falls within `separation`
    idxs = target_coord.separation(coords) < separation * u.arcsec
    if idxs.sum() > 0:
        # result may be multiple objects
        res = df[idxs]
        if verbose:
            targets = res["Target"].values
            print(f"There are {len(res)} matches: {targets}")
            print(f"{df.loc[idxs, df.columns[7:14]].T}\n\n")
        return res

    else:
        # find the nearest HARPS object in the database to target
        # idx, sep2d, dist3d = match_coordinates_3d(
        #     target_coord, coords, nthneighbor=1)
        idx = target_coord.separation(coords).argmin()
        sep2d = target_coord.separation(coords[idx])
        nearest_obj = df.iloc[idx]["Target"]
        ra, dec = df.iloc[idx][["RA", "DEC"]]
        print(
            f"Nearest HARPS object is\n{nearest_obj}: ra,dec=({ra},{dec}) @ d={sep2d.arcsec/60:.2f} arcmin\n"
        )
        return None


# def get_harps_bank(url, verbose=True):
#     """
#     Download archival HARPS data from url
#     http://www.mpia.de/homes/trifonov/HARPS_RVBank.html
#     """
#     homeurl = ""
#     fp = os.path.join(outdir, "HARPS_RVBank_table.csv")
#     return


def get_mamajek_table(clobber=False, verbose=True, data_loc=DATA_PATH):
    fp = join(data_loc, "mamajek_table.csv")
    if not exists(fp) or clobber:
        url = "http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt"
        # cols="SpT Teff logT BCv Mv logL B-V Bt-Vt G-V U-B V-Rc V-Ic V-Ks J-H H-Ks Ks-W1 W1-W2 W1-W3 W1-W4 Msun logAge b-y M_J M_Ks Mbol i-z z-Y R_Rsun".split(' ')
        df = pd.read_csv(
            url,
            skiprows=21,
            skipfooter=524,
            delim_whitespace=True,
            engine="python",
        )
        # tab = ascii.read(url, guess=None, data_start=0, data_end=124)
        # df = tab.to_pandas()
        # replace ... with NaN
        df = df.replace(["...", "....", "....."], np.nan)
        # replace header
        # df.columns = cols
        # drop last duplicate column
        df = df.drop(df.columns[-1], axis=1)
        # df['#SpT_num'] = range(df.shape[0])
        # df['#SpT'] = df['#SpT'].astype('category')

        # remove the : type in M_J column
        df["M_J"] = df["M_J"].apply(lambda x: str(x).split(":")[0])
        # convert columns to float
        for col in df.columns:
            if col == "#SpT":
                df[col] = df[col].astype("category")
            else:
                df[col] = df[col].astype(float)
            # if col=='SpT':
            #     df[col] = df[col].astype('categorical')
            # else:
            #     df[col] = df[col].astype(float)
        df.to_csv(fp, index=False)
        print(f"Saved: {fp}")
    else:
        df = pd.read_csv(fp)
        if verbose:
            print(f"Loaded: {fp}")
    return df


def get_mag_err_from_flux(flux, flux_err):
    """
    equal to 1.086/(S/N)
    """
    return 2.5 * np.log10(1 + flux_err / flux)


def get_absolute_gmag(gmag, distance, a_g):
    """
    gmag : float
        apparent G band magnitude
    distance : float
        distance in pc
    a_g : float
        extinction in the G-band
    """
    assert (gmag is not None) & (str(gmag) != "nan"), "gma is nan"
    assert (distance is not None) & (str(distance) != "nan"), "distance is nan"
    assert (a_g is not None) & (str(a_g) != "nan"), "a_g is nan"
    Gmag = gmag - 5.0 * np.log10(distance) + 5.0 - a_g
    return Gmag


def get_excess_from_extiction(A_g, color="bp_rp"):
    """
    Compute excess from difference in extinctions E(Bp-Rp) = A_Bp-A_Rp
    using coefficients from Malhan, Ibata & Martin (2018a)
    and extinction in G-band A_g

    Compare the result to 'e_bp_min_rp_val' column in gaia table
    which is the estimate of redenning E[BP-RP] from Apsis-Priam.
    """
    assert A_g is not None
    assert str(A_g) != "nan"
    # ratio of A_X/A_V
    if color == "bp_rp":
        # E(Bp-Rp) = A_Bp-A_Rp
        Ag_Av = extinction_ratios["G"]
        Ab_Av = extinction_ratios["Bp"]
        Ar_Av = extinction_ratios["Rp"]
        Ab_minus_Ar = (A_g / Ag_Av) * (Ab_Av - Ar_Av)  # difference
    else:
        errmsg = "color=bp_rp is only implemented"
        raise NotImplementedError(errmsg)
    return Ab_minus_Ar


def get_absolute_color_index(A_g, bmag, rmag):
    """
    Deredden the Gaia Bp-Rp color using Bp-Rp extinction ratio (==Bp-Rp excess)

    E(Bp-Rp) = A_Bp - A_Rp = (Bp-Rp)_obs - (Bp-Rp)_abs
    --> (Bp-Rp)_abs = (Bp-Rp)_obs - E(Bp-Rp)

    Note that 'bmag-rmag' is same as bp_rp column in gaia table
    See also http://www.astro.ncu.edu.tw/~wchen/Courses/ISM/11.Extinction.pdf
    """
    assert (A_g is not None) & (str(A_g) != "nan")
    assert (bmag is not None) & (str(bmag) != "nan")
    assert (rmag is not None) & (str(rmag) != "nan")
    # E(Bp-Rp) = A_Bp-A_Rp = (Bp-Rp)_obs - E(Bp-Rp)
    Ab_minus_Ar = get_excess_from_extiction(A_g)
    bp_rp = bmag - rmag  # color index
    Bp_Rp = bp_rp - Ab_minus_Ar
    return Bp_Rp


def get_distance(m, M, Av=0):
    """
    calculate distance [in pc] from extinction-corrected magnitude
    using the equation: d=10**((m-M+5-Av)/5)

    Note: m-M=5*log10(d)-5+Av
    see http://astronomy.swin.edu.au/cosmos/I/Interstellar+Reddening

    Parameters
    ---------
    m : apparent magnitude
    M : absolute magnitude
    Av : extinction (in V band)
    """
    assert (m is not None) & (str(m) != "nan")
    assert (M is not None) & (str(M) != "nan")
    distance = 10 ** (0.2 * (m - M + 5 - Av))
    return distance


def parse_aperture_mask(
    tpf,
    sap_mask="pipeline",
    aper_radius=None,
    percentile=None,
    verbose=False,
    threshold_sigma=None,
):
    """Parse and make aperture mask"""
    if verbose:
        if sap_mask == "round":
            print(
                "aperture photometry mask: {} (r={} pix)\n".format(
                    sap_mask, aper_radius
                )
            )
        elif sap_mask == "square":
            print(
                "aperture photometry mask: {0} ({1}x{1} pix)\n".format(
                    sap_mask, aper_radius
                )
            )
        elif sap_mask == "percentile":
            print(
                "aperture photometry mask: {} ({}%)\n".format(
                    sap_mask, percentile
                )
            )
        else:
            print("aperture photometry mask: {}\n".format(sap_mask))

    # stacked_img = np.median(tpf.flux,axis=0)
    if (sap_mask == "pipeline") or (sap_mask is None):
        errmsg = "tpf does not have pipeline mask"
        assert tpf.pipeline_mask is not None, errmsg
        mask = tpf.pipeline_mask  # default
    elif sap_mask == "all":
        mask = np.ones((tpf.shape[1], tpf.shape[2]), dtype=bool)
    elif sap_mask == "round":
        assert aper_radius is not None, "supply aper_radius"
        mask = make_round_mask(tpf.flux[0], radius=aper_radius)
    elif sap_mask == "square":
        assert aper_radius is not None, "supply aper_radius/size"
        mask = make_square_mask(tpf.flux[0], size=aper_radius, angle=None)
    elif sap_mask == "threshold":
        assert threshold_sigma is not None, "supply threshold_sigma"
        # FIXME: make sure aperture is contiguous
        mask = tpf.create_threshold_mask(threshold_sigma)
    elif sap_mask == "percentile":
        assert percentile is not None, "supply percentile"
        median_img = np.nanmedian(tpf.flux, axis=0)
        mask = median_img > np.nanpercentile(median_img, percentile)
    else:
        raise ValueError("Unknown aperture mask")
    return mask


def make_round_mask(img, radius, xy_center=None):
    """Make round mask in units of pixels

    Parameters
    ----------
    img : numpy ndarray
        image
    radius : int
        aperture mask radius or size
    xy_center : tuple
        aperture mask center position

    Returns
    -------
    mask : np.ma.masked_array
        aperture mask
    """
    offset = 2  # from center
    xcen, ycen = img.shape[0] // 2, img.shape[1] // 2
    if xy_center is None:  # use the middle of the image
        y, x = np.unravel_index(np.argmax(img), img.shape)
        xy_center = [x, y]
        # check if near edge
        if np.any([abs(x - xcen) > offset, abs(y - ycen) > offset]):
            print("Brightest star is detected far from the center.")
            print("Aperture mask is placed at the center instead.\n")
            xy_center = [xcen, ycen]

    Y, X = np.ogrid[: img.shape[0], : img.shape[1]]
    dist_from_center = np.sqrt(
        (X - xy_center[0]) ** 2 + (Y - xy_center[1]) ** 2
    )

    mask = dist_from_center <= radius
    return np.ma.masked_array(img, mask=mask).mask


def make_square_mask(img, size, xy_center=None, angle=None):
    """Make rectangular mask with optional rotation

    Parameters
    ----------
    img : numpy ndarray
        image
    size : int
        aperture mask size
    xy_center : tuple
        aperture mask center position
    angle : int
        rotation

    Returns
    -------
    mask : np.ma.masked_array
        aperture mask
    """
    offset = 2  # from center
    xcen, ycen = img.shape[0] // 2, img.shape[1] // 2
    if xy_center is None:  # use the middle of the image
        y, x = np.unravel_index(np.argmax(img), img.shape)
        xy_center = [x, y]
        # check if near edge
        if np.any([abs(x - xcen) > offset, abs(y - ycen) > offset]):
            print("Brightest star detected is far from the center.")
            print("Aperture mask is placed at the center instead.\n")
            xy_center = [xcen, ycen]
    mask = np.zeros_like(img, dtype=bool)
    mask[ycen - size : ycen + size + 1, xcen - size : xcen + size + 1] = True
    # if angle:
    #    #rotate mask
    #    mask = rotate(mask, angle, axes=(1, 0), reshape=True, output=bool, order=0)
    return mask


def remove_bad_data(tpf, sector=None, verbose=True):
    """Remove bad cadences identified in data release notes

    https://arxiv.org/pdf/2003.10451.pdf, S4.5:
    all transiting planets with periods 10.5-17.5 d could be
    hidden by the masking in the PDC light curves if only
    observed in Sector 14.

    Parameters
    ----------
    tpf : lk.targetpixelfile

    sector : int
        TESS sector
    verbose : bool
        print texts
    """
    if sector is None:
        sector = tpf.sector
    if verbose:
        print(
            f"Applying data quality mask identified in Data Release Notes (sector {sector}):"
        )
    if sector == 1:
        pointing_jitter_start = 1346
        pointing_jitter_end = 1350
        if verbose:
            print(
                "t<{}|t>{}\n".format(
                    pointing_jitter_start, pointing_jitter_end
                )
            )
        tpf = tpf[
            (tpf.time < pointing_jitter_start)
            | (tpf.time > pointing_jitter_end)
        ]
    if sector == 2:
        if verbose:
            print("None.\n")
    if sector == 3:
        science_data_start = 1385.89663
        science_data_end = 1406.29247
        if verbose:
            print("t>{}|t<{}\n".format(science_data_start, science_data_end))
        tpf = tpf[
            (tpf.time > science_data_start) | (tpf.time < science_data_end)
        ]
    if sector == 4:
        guidestar_tables_replaced = 1413.26468
        instru_anomaly_start = 1418.53691
        data_collection_resumed = 1421.21168
        if verbose:
            print(
                "t>{}|t<{}|t>{}\n".format(
                    guidestar_tables_replaced,
                    instru_anomaly_start,
                    data_collection_resumed,
                )
            )
        tpf = tpf[
            (tpf.time > guidestar_tables_replaced)
            | (tpf.time < instru_anomaly_start)
            | (tpf.time > data_collection_resumed)
        ]
    if sector == 5:
        # use of Cam1 in attitude control was disabled for the
        # last ~0.5 days of orbit due to o strong scattered light
        cam1_guide_disabled = 1463.93945
        if verbose:
            print("t<{}\n".format(cam1_guide_disabled))
        tpf = tpf[tpf.time < cam1_guide_disabled]
    if sector == 6:
        # ~3 days of orbit 19 were used to collect calibration
        # data for measuring the PRF of cameras;
        # reaction wheel speeds were reset with momentum dumps
        # every 3.125 days
        data_collection_start = 1468.26998
        if verbose:
            print("t>{}\n".format(data_collection_start))
        tpf = tpf[tpf.time > data_collection_start]
    if sector == 8:
        # interruption in communications between instru and spacecraft occurred
        cam1_guide_enabled = 1517.39566
        orbit23_end = 1529.06510
        cam1_guide_enabled2 = 1530.44705
        instru_anomaly_start = 1531.74288
        data_colletion_resumed = 1535.00264
        if verbose:
            print(
                "t>{}|t<{}|t>{}|t<{}|t>{}\n".format(
                    cam1_guide_enabled,
                    orbit23_end,
                    cam1_guide_enabled2,
                    instru_anomaly_start,
                    data_colletion_resumed,
                )
            )
        tpf = tpf[
            (tpf.time > cam1_guide_enabled)
            | (tpf.time <= orbit23_end)
            | (tpf.time > cam1_guide_enabled2)
            | (tpf.time < instru_anomaly_start)
            | (tpf.time > data_colletion_resumed)
        ]
    if sector == 9:
        """


        use of Cam1 in attitude control was disabled at the
        start of both orbits due to strong scattered light"""
        cam1_guide_enabled = 1543.75080
        orbit25_end = 1555.54148
        cam1_guide_enabled2 = 1543.75080
        if verbose:
            print(
                "t>{}|t<{}|t>{}\n".format(
                    cam1_guide_enabled, orbit25_end, cam1_guide_enabled2
                )
            )
        tpf = tpf[
            (tpf.time > cam1_guide_enabled)
            | (tpf.time <= orbit25_end)
            | (tpf.time > cam1_guide_enabled2)
        ]
    if sector == 10:
        """
        https://archive.stsci.edu/missions/tess/doc/tess_drn/tess_sector_10_drn14_v02.pdf

        Total of 25.27 days of science data collected

        use of Cam1 in attitude control was disabled at the
        start of both orbits due to strong scattered light
        """
        cam1_guide_enabled = 1570.87620
        orbit27_end = 1581.78453
        cam1_guide_enabled2 = 1584.72342
        if verbose:
            print(
                "t>{}|t<{}|t>{}\n".format(
                    cam1_guide_enabled, orbit27_end, cam1_guide_enabled2
                )
            )
        tpf = tpf[
            (tpf.time > cam1_guide_enabled)
            | (tpf.time <= orbit27_end)
            | (tpf.time > cam1_guide_enabled2)
        ]
    if sector == 11:
        """
        https://archive.stsci.edu/missions/tess/doc/tess_drn/tess_sector_11_drn16_v02.pdf

        use of Cam1 in attitude control was disabled at the
        start of both orbits due to strong scattered light

        Total of 26.04 days of science data collected
        """
        cam1_guide_enabled = 1599.94148
        orbit29_end = 1609.69425
        cam1_guide_enabled2 = 1614.19842
        if verbose:
            print(
                "t>{}|t<{}|t>{}\n".format(
                    cam1_guide_enabled, orbit29_end, cam1_guide_enabled2
                )
            )
        tpf = tpf[
            (tpf.time > cam1_guide_enabled)
            | (tpf.time <= orbit29_end)
            | (tpf.time > cam1_guide_enabled2)
        ]

    if sector in [12, 13, 14, 15, 16, 17, 19, 20, 21]:
        """
        See list of release notes:
        http://archive.stsci.edu/tess/tess_drn.html

        Total days of science data collected:
        12: 26.90
        13: 27.51
        14: 25.91
        15: 24.97
        16: 23.38
        17: 23.51
        19: 24.10
        20: 24.79
        21: 24.42

        Note on sector 14:
        * first northern ecliptic hemisphere pointing
        * first sector to make use of TIC 8 based on Gaia DR2 astrometry+photometry
        * spacecraft is pointed to a higher ecliptic latitude (+85 degrees rather
        than +54 degrees) to mitigate issues with scattered light in Cam 1 and Cam 2
        * first to make use of an updated SPOC data processing
        pipeline, SPOC Release 4.0
        * the first to make use of CCD-specific Data Anomaly Flags that mark
        cadences excluded due to high levels of scattered light. The flags are referred to as
        “Scattered Light” flags and marked with bit 13, value 4096
        """
        print(f"No instrument anomaly in sector {sector}")

    if sector == 18:
        """
        * spacecraft passed through the shadow of the Earth at the start of orbit 43
        during which the instrument was turned off and no data were collected for 6.2 hr
        * thermal state of the spacecraft changed during this time,
        and trends in the raw photometry and target positions are apparent after data collection
        resumed

        Total of 23.12 days of science data collected
        """
        instru_restart = 1791.36989
        orbit43_end = 1802.43999
        if verbose:
            print("t>{}|t<{}\n".format(instru_restart, orbit43_end))
        tpf = tpf[(tpf.time > instru_restart) | (tpf.time <= orbit29_end)]

    return tpf


def get_tois(
    clobber=True,
    outdir=DATA_PATH,
    verbose=False,
    remove_FP=True,
    remove_known_planets=False,
    add_FPP=False,
):
    """Download TOI list from TESS Alert/TOI Release.

    Parameters
    ----------
    clobber : bool
        re-download table and save as csv file
    outdir : str
        download directory location
    verbose : bool
        print texts

    Returns
    -------
    d : pandas.DataFrame
        TOI table as dataframe
    """
    dl_link = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    fp = join(outdir, "TOIs.csv")
    if not exists(outdir):
        os.makedirs(outdir)

    if not exists(fp) or clobber:
        d = pd.read_csv(dl_link)  # , dtype={'RA': float, 'Dec': float})
        msg = f"Downloading {dl_link}\n"
        if add_FPP:
            fp2 = join(outdir, "Giacalone2020/tab4.txt")
            classified = ascii.read(fp2).to_pandas()
            fp3 = join(outdir, "Giacalone2020/tab5.txt")
            unclassified = ascii.read(fp3).to_pandas()
            fpp = pd.concat(
                [
                    classified[["TOI", "FPP-2m", "FPP-30m"]],
                    unclassified[["TOI", "FPP"]],
                ],
                sort=True,
            )
            d = pd.merge(d, fpp, how="outer").drop_duplicates()
        d.to_csv(fp, index=False)
    else:
        d = pd.read_csv(fp).drop_duplicates()
        msg = f"Loaded: {fp}\n"
    assert len(d) > 1000, f"{fp} likely has been overwritten!"

    # remove False Positives
    if remove_FP:
        d = d[d["TFOPWG Disposition"] != "FP"]
        msg += "TOIs with TFPWG disposition==FP are removed.\n"
    if remove_known_planets:
        planet_keys = [
            "HD",
            "GJ",
            "LHS",
            "XO",
            "Pi Men" "WASP",
            "SWASP",
            "HAT",
            "HATS",
            "KELT",
            "TrES",
            "QATAR",
            "CoRoT",
            "K2",  # , "EPIC"
            "Kepler",  # "KOI"
        ]
        keys = []
        for key in planet_keys:
            idx = ~np.array(
                d["Comments"].str.contains(key).tolist(), dtype=bool
            )
            d = d[idx]
            if idx.sum() > 0:
                keys.append(key)
        msg += f"{keys} planets are removed.\n"
    msg += f"Saved: {fp}\n"
    if verbose:
        print(msg)
    return d.sort_values("TOI")


def get_toi(toi, verbose=False, remove_FP=True, clobber=False):
    """Query TOI from TOI list

    Parameters
    ----------
    toi : float
        TOI id
    clobber : bool
        re-download csv file
    outdir : str
        csv path
    verbose : bool
        print texts

    Returns
    -------
    q : pandas.DataFrame
        TOI match else None
    """
    df = get_tois(verbose=False, remove_FP=remove_FP, clobber=clobber)

    if isinstance(toi, int):
        toi = float(str(toi) + ".01")
    else:
        planet = str(toi).split(".")[1]
        assert len(planet) == 2, "use pattern: TOI.01"
    idx = df["TOI"].isin([toi])
    q = df.loc[idx]
    assert len(q) > 0, "TOI not found!"

    q.index = q["TOI"].values
    if verbose:
        print("Data from TOI Release:\n")
        columns = [
            "Period (days)",
            "Epoch (BJD)",
            "Duration (hours)",
            "Depth (ppm)",
            "Comments",
        ]
        print(f"{q[columns].T}\n")

    if q["TFOPWG Disposition"].isin(["FP"]).any():
        print("\nTFOPWG disposition is a False Positive!\n")

    return q.sort_values(by="TOI", ascending=True)


def get_ctois(clobber=True, outdir=DATA_PATH, verbose=False, remove_FP=True):
    """Download Community TOI list from exofop/TESS.

    Parameters
    ----------
    clobber : bool
        re-download table and save as csv file
    outdir : str
        download directory location
    verbose : bool
        print texts

    Returns
    -------
    d : pandas.DataFrame
        CTOI table as dataframe

    See interface: https://exofop.ipac.caltech.edu/tess/view_ctoi.php
    See also: https://exofop.ipac.caltech.edu/tess/ctoi_help.php
    """
    dl_link = "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv"
    fp = join(outdir, "CTOIs.csv")
    if not exists(outdir):
        os.makedirs(outdir)

    if not exists(fp) or clobber:
        d = pd.read_csv(dl_link)  # , dtype={'RA': float, 'Dec': float})
        msg = "Downloading {}\n".format(dl_link)
    else:
        d = pd.read_csv(fp).drop_duplicates()
        msg = "Loaded: {}\n".format(fp)
    d.to_csv(fp, index=False)

    # remove False Positives
    if remove_FP:
        d = d[d["User Disposition"] != "FP"]
        msg += "CTOIs with user disposition==FP are removed.\n"
    msg += "Saved: {}\n".format(fp)
    if verbose:
        print(msg)
    return d.sort_values("CTOI")


def get_ctoi(ctoi, verbose=False, remove_FP=False, clobber=False):
    """Query CTOI from CTOI list

    Parameters
    ----------
    ctoi : float
        CTOI id

    Returns
    -------
    q : pandas.DataFrame
        CTOI match else None
    """
    ctoi = float(ctoi)
    df = get_ctois(verbose=False, remove_FP=remove_FP, clobber=clobber)

    if isinstance(ctoi, int):
        ctoi = float(str(ctoi) + ".01")
    else:
        planet = str(ctoi).split(".")[1]
        assert len(planet) == 2, "use pattern: CTOI.01"
    idx = df["CTOI"].isin([ctoi])

    q = df.loc[idx]
    assert len(q) > 0, "CTOI not found!"

    q.index = q["CTOI"].values
    if verbose:
        print("Data from CTOI Release:\n")
        columns = [
            "Period (days)",
            "Midpoint (BJD)",
            "Duration (hours)",
            "Depth ppm",
            "Notes",
        ]
        print(f"{q[columns].T}\n")
    if (q["TFOPWG Disposition"].isin(["FP"]).any()) | (
        q["User Disposition"].isin(["FP"]).any()
    ):
        print("\nTFOPWG/User disposition is a False Positive!\n")

    return q.sort_values(by="CTOI", ascending=True)


def get_specs_table_from_tfop(clobber=True, outdir=DATA_PATH, verbose=True):
    """
    html:
    https://exofop.ipac.caltech.edu/tess/view_spect.php?sort=id&ipp1=1000

    plot notes:
    https://exofop.ipac.caltech.edu/tess/classification_plots.php
    """
    base = "https://exofop.ipac.caltech.edu/tess/"
    fp = os.path.join(outdir, "tfop_sg2_spec_table.csv")
    if not os.path.exists(fp) or clobber:
        url = base + "download_spect.php?sort=id&output=csv"
        df = pd.read_csv(url)
        df.to_csv(fp, index=False)
        if verbose:
            print(f"Saved: {fp}")
    else:
        df = pd.read_csv(fp)
        if verbose:
            print(f"Loaded: {fp}")
    return df


def get_target_coord(
    ra=None,
    dec=None,
    toi=None,
    ctoi=None,
    tic=None,
    epic=None,
    gaiaid=None,
    name=None,
):
    """get target coordinate
    """
    if np.all([ra, dec]):
        target_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    elif toi:
        toi_params = get_toi(toi=toi, clobber=False, verbose=False)
        target_coord = SkyCoord(
            ra=toi_params["RA"].values[0],
            dec=toi_params["Dec"].values[0],
            distance=toi_params["Stellar Distance (pc)"].values[0],
            unit=(u.hourangle, u.degree, u.pc),
        )
    elif ctoi:
        ctoi_params = get_ctoi(ctoi=ctoi, clobber=False, verbose=False)
        target_coord = SkyCoord(
            ra=ctoi_params["RA"].values[0],
            dec=ctoi_params["Dec"].values[0],
            distance=ctoi_params["Stellar Distance (pc)"].values[0],
            unit=(u.degree, u.degree, u.pc),
        )
    elif tic:
        df = Catalogs.query_criteria(catalog="Tic", ID=tic).to_pandas()
        target_coord = SkyCoord(
            ra=df.iloc[0]["ra"],
            dec=df.iloc[0]["dec"],
            distance=Distance(parallax=df.iloc[0]["plx"] * u.mas).pc,
            unit=(u.degree, u.degree, u.pc),
        )
    # name resolver
    elif epic is not None:
        try:
            import k2plr

            client = k2plr.API()
        except Exception:
            raise ModuleNotFoundError(
                "pip install git+https://github.com/rodluger/k2plr.git"
            )
        try:
            target_coord = SkyCoord.from_name(f"EPIC {epic}")
        except Exception:
            star = client.k2_star(int(epic))
            ra = float(star.k2_ra)
            dec = float(star.k2_dec)
            target_coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    elif gaiaid is not None:
        target_coord = SkyCoord.from_name(f"Gaia DR2 {gaiaid}")
    elif name is not None:
        target_coord = SkyCoord.from_name(name)
    else:
        raise ValueError("Supply RA & Dec, TOI, TIC, or Name")
    return target_coord


def parse_target_coord(target):
    """
    parse target string and query coordinates; e.g.
    toi.X, ctoi.X, tic.X, gaiaX, epicX, Simbad name
    """
    assert isinstance(target, str)
    if len(target.split(",")) == 2:
        # coordinates: ra, dec
        if len(target.split(":")) == 6:
            # e.g. 01:02:03.0, 04:05:06.0
            coord = SkyCoord(target, unit=("hourangle", "degree"))
        else:
            # e.g. 70.5, 80.5
            coord = SkyCoord(target, unit=("degree", "degree"))
    else:
        # name or ID
        if target[:3] == "toi":
            toiid = float(target[3:])
            coord = get_coord_from_toiid(toiid)
        elif target[:4] == "ctoi":
            ctoiid = float(target[4:])
            coord = get_coord_from_ctoiid(ctoiid)
        elif target[:3] == "tic":
            # TODO: requires int for astroquery.mast.Catalogs to work
            if len(target[3:].split(".")) == 2:
                ticid = int(target[3:].split(".")[1])
            else:
                ticid = int(target[3:])
            coord = get_coord_from_ticid(ticid)
        elif (
            (target[:4] == "epic")
            | (target[:2] == "k2")
            | (target[:4] == "gaia")
        ):
            # coord = get_coord_from_epicid(epicid)
            coord = SkyCoord.from_name(target)
        else:
            coord = SkyCoord.from_name(target)
    return coord


def get_epicid_from_k2name(k2name):
    res = lk.search_targetpixelfile(k2name, mission="K2")
    target_name = res.table.to_pandas().target_name[0]
    epicid = int(target_name[4:])  # skip ktwo
    return epicid


def get_coord_from_toiid(toiid, **kwargs):
    toi = get_toi(toiid, **kwargs)
    coord = SkyCoord(
        ra=toi["RA"].values[0],
        dec=toi["Dec"].values[0],
        distance=toi["Stellar Distance (pc)"].values[0],
        unit=(u.hourangle, u.degree, u.pc),
    )
    return coord


def get_coord_from_ctoiid(ctoiid, **kwargs):
    ctoi = get_ctoi(ctoiid, **kwargs)
    coord = SkyCoord(
        ra=ctoi["RA"].values[0],
        dec=ctoi["Dec"].values[0],
        distance=ctoi["Stellar Distance (pc)"].values[0],
        unit=(u.degree, u.degree, u.pc),
    )
    return coord


def get_coord_from_ticid(ticid):
    df = Catalogs.query_criteria(catalog="Tic", ID=ticid).to_pandas()
    coord = SkyCoord(
        ra=df.iloc[0]["ra"],
        dec=df.iloc[0]["dec"],
        distance=Distance(parallax=df.iloc[0]["plx"] * u.mas).pc,
        unit=(u.degree, u.degree, u.pc),
    )
    return coord


def get_coord_from_epicid(epicid):
    try:
        import k2plr

        client = k2plr.API()
    except Exception:
        raise ModuleNotFoundError(
            "pip install git+https://github.com/rodluger/k2plr.git"
        )
    epicid = int(epicid)
    star = client.k2_star(epicid)
    ra = float(star.k2_ra)
    dec = float(star.k2_dec)
    coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    return coord


def get_target_coord_3d(target_coord, verbose=False):
    """append distance to target coordinate"""
    if verbose:
        print("Querying parallax of target from Gaia\n")
    g = Gaia.query_object(target_coord, radius=10 * u.arcsec).to_pandas()
    gcoords = SkyCoord(ra=g["ra"], dec=g["dec"], unit="deg")
    # FIXME: get minimum or a few stars around the minimum?
    idx = target_coord.separation(gcoords).argmin()
    star = g.loc[idx]
    # get distance from parallax
    target_dist = Distance(parallax=star["parallax"].values * u.mas)
    # redefine skycoord with coord and distance
    target_coord = SkyCoord(
        ra=target_coord.ra, dec=target_coord.dec, distance=target_dist
    )
    return target_coord


def get_toi_coord_3d(toi, clobber=False, verbose=False):
    all_tois = get_tois(clobber=clobber, verbose=verbose)
    idx = all_tois["TOI"].isin([toi])
    columns = ["RA", "Dec", "Stellar Distance (pc)"]
    ra, dec, dist = all_tois.loc[idx, columns].values[0]
    target_coord = SkyCoord(
        ra=ra, dec=dec, distance=dist, unit=(u.hourangle, u.deg, u.pc)
    )
    return target_coord


def get_transformed_coord(df, frame="galactocentric", verbose=True):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        catalog with complete kinematics parameters
    frame : str
        frame conversion

    Returns
    -------
    df : pandas.DataFrame
        catalog with transformed coordinates appended in columns

    Note
    ----
    Assumes galactic center distance distance of 8.1 kpc based on the GRAVITY
    collaboration, and a solar height of z_sun=0 pc.
    See also:
    http://learn.astropy.org/rst-tutorials/gaia-galactic-orbits.html?highlight=filtertutorials
    """
    assert len(df) > 0, "df is empty"
    if np.any(df["parallax"] < 0):
        # retain non-negative parallaxes including nan
        df = df[(df["parallax"] >= 0) | (df["parallax"].isnull())]
        if verbose:
            print("Some parallaxes are negative!")
            print("These are removed for the meantime.")
            print("For proper treatment, see:")
            print("https://arxiv.org/pdf/1804.09366.pdf\n")

    icrs = SkyCoord(
        ra=df["ra"].values * u.deg,
        dec=df["dec"].values * u.deg,
        distance=Distance(parallax=df["parallax"].values * u.mas),
        radial_velocity=df["radial_velocity"].values * u.km / u.s,
        pm_ra_cosdec=df["pmra"].values * u.mas / u.yr,
        pm_dec=df["pmdec"].values * u.mas / u.yr,
        frame="fk5",
        equinox="J2000.0",
    )
    # transform to galactocentric frame
    if frame == "galactocentric":
        xyz = icrs.transform_to(
            Galactocentric(z_sun=0 * u.pc, galcen_distance=8.1 * u.kpc)
        )
        xyz = icrs.galactocentric
        df["X"] = xyz.x.copy()
        df["Y"] = xyz.y.copy()
        df["Z"] = xyz.z.copy()
        df["U"] = xyz.v_x.copy()
        df["V"] = xyz.v_y.copy()
        df["W"] = xyz.v_z.copy()

    elif frame == "galactic":
        # transform to galactic frame
        gal = icrs.transform_to("galactic")
        df["gal_l"] = gal.l.deg.copy()
        df["gal_b"] = gal.b.deg.copy()
        df["gal_pm_b"] = gal.pm_b.copy()
        df["gal_pm_l_cosb"] = gal.pm_l_cosb.copy()
    else:
        raise ValueError("frame unavailable")
    return df


def query_gaia_params_of_all_tois(
    fp=None, verbose=True, clobber=False, update=True
):
    """
    See also
    https://astroquery.readthedocs.io/en/latest/xmatch/xmatch.html

    Note: Ticv8 is preferable since it has Gaia DR2 parameters and more
    See: https://mast.stsci.edu/api/v0/pyex.html#MastTicCrossmatchPy
    """
    if fp is None:
        fp = join(DATA_PATH, "toi_gaia_params.hdf5")

    tois = get_tois(verbose=verbose, clobber=clobber)
    toiids = np.unique(tois.TOI.astype(float))
    if not exists(fp) or clobber:
        # download all from gaia catalog
        if verbose:
            print(f"Querying Gaia DR2 catalog for {len(toiids)} TOIs\n")
        toi_gaia_params = {}
        for toi in tqdm(toiids):
            try:
                t = target.Target(toiid=toi, verbose=False)
                # query gaia dr2 catalog to get gaia id
                df_gaia = t.query_gaia_dr2_catalog(return_nearest_xmatch=True)
                # t.target_coord.distance = Distance(parallax=df_gaia.parallax*u.mas)
                toi_gaia_params[toi] = df_gaia
            except Exception as e:
                if verbose:
                    print(e)
        dd.io.save(fp, toi_gaia_params)
        msg = f"Saved: {fp}"
    elif exists(fp) and update:
        # load file and append new queries
        if verbose:
            print("Querying Gaia DR2 catalog for new TOIs\n")
        toi_gaia_params = dd.io.load(fp)
        downloaded_tois = np.sort(list(toi_gaia_params.keys()))
        for toi in tqdm(toiids):
            if toi > downloaded_tois[-1]:
                try:
                    t = target.Target(toiid=toi, verbose=verbose)
                    # query gaia dr2 catalog to get gaia id
                    df_gaia = t.query_gaia_dr2_catalog(
                        return_nearest_xmatch=True
                    )
                    # update
                    toi_gaia_params.update({toi: df_gaia})
                except Exception as e:
                    if verbose:
                        print(e)
        dd.io.save(fp, toi_gaia_params)
        msg = f"Saved: {fp}"
    else:
        # load
        toi_gaia_params = dd.io.load(fp)
        msg = f"Loaded: {fp}"
    if verbose:
        print(msg)
    # convert dict of series into a single df
    sample = list(toi_gaia_params.values())[0]  # any would do
    if isinstance(sample, pd.Series):
        df = pd.concat(toi_gaia_params, axis=1, ignore_index=False).T
    # convert dict of df into a single df
    else:
        df = pd.concat(toi_gaia_params.values(), ignore_index=True)

    df.index.name = "TOI"
    return df


# def get_K2_targetlist(campaign, outdir=DATA_PATH, verbose=True):
#     """
#     campaign: K2 campaign number [0-18]
#     """
#     if verbose:
#         print("Retrieving K2 campaign {} target list...\n".format(campaign))
#
#     file_list = sorted(glob(os.path.join(outdir, "*csv")))
#
#     if len(file_list) == 0:
#         link = (
#             "https://keplerscience.arc.nasa.gov/data/campaigns/c"
#             + str(campaign)
#             + "/K2Campaign"
#             + str(campaign)
#             + "targets.csv"
#         )
#         d = pd.read_csv(link)
#         d = clean_df(d)
#         if not os.path.exists(outdir):
#             os.makedirs(outdir)
#         name = link.split("/"[-1])
#         outpath = os.path.join(outdir, name)
#         targets.to_csv(outpath)
#     else:
#         fp = os.path.join(outdir, "K2Campaign" + str(campaign) + "targets.csv")
#
#         dtypes = {
#             "EPIC": int,
#             "RA": float,
#             "Dec": float,
#             "Kp": float,
#             "InvestigationIDs": str,
#         }
#         d = pd.read_csv(fp, delimiter=",", skipinitialspace=True, dtype=dtypes)
#         targets = clean_df(d)
#
#     # targets = targets.replace(r'^\s+$', np.nan, regex=True)
#     return targets


def get_cartersian_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def is_point_inside_mask(border, target):
    """determine if target coordinate is within polygon border
    """
    degree = 0
    for i in range(len(border) - 1):
        a = border[i]
        b = border[i + 1]

        # calculate distance of vector
        A = get_cartersian_distance(a[0], a[1], b[0], b[1])
        B = get_cartersian_distance(target[0], target[1], a[0], a[1])
        C = get_cartersian_distance(target[0], target[1], b[0], b[1])

        # calculate direction of vector
        ta_x = a[0] - target[0]
        ta_y = a[1] - target[1]
        tb_x = b[0] - target[0]
        tb_y = b[1] - target[1]

        cross = tb_y * ta_x - tb_x * ta_y
        clockwise = cross < 0

        # calculate sum of angles
        if clockwise:
            degree = degree + np.rad2deg(
                np.arccos((B * B + C * C - A * A) / (2.0 * B * C))
            )
        else:
            degree = degree - np.rad2deg(
                np.arccos((B * B + C * C - A * A) / (2.0 * B * C))
            )

    if abs(round(degree) - 360) <= 3:
        return True
    return False


def PadWithZeros(vector, pad_width, iaxis, kwargs):
    vector[: pad_width[0]] = 0
    vector[-pad_width[1] :] = 0
    return vector


def get_fluxes_within_mask(tpf, aper_mask, gaia_sources):
    """compute relative fluxes of gaia sources within aperture
    To compute the actual depth taking into account dilution,
    delta_true = delta_obs*gamma, where
    gamma = 1+10**(0.4*dmag) [dilution factor]
    """
    assert tpf is not None
    assert aper_mask is not None
    assert gaia_sources is not None
    ra, dec = gaia_sources[["ra", "dec"]].values.T
    pix_coords = tpf.wcs.all_world2pix(np.c_[ra, dec], 0)
    contour_points = measure.find_contours(aper_mask, level=0.1)[0]
    isinside = [
        is_point_inside_mask(contour_points, pix) for pix in pix_coords
    ]
    min_gmag = gaia_sources.loc[isinside, "phot_g_mean_mag"].min()
    gamma = gaia_sources.loc[isinside, "phot_g_mean_mag"].apply(
        lambda x: 10 ** (0.4 * (min_gmag - x))
    )
    return gamma


def get_limbdark(
    band, teff=None, logg=None, feh=None, tic_params=None, **kwargs
):
    """
    """
    try:
        import limbdark as ld
    except Exception:
        command = (
            "pip install git+https://github.com/john-livingston/limbdark.git"
        )
        raise ModuleNotFoundError(command)

    if (teff is None) & (logg is None) & (feh is None):
        teff = (tic_params["Teff"], tic_params["e_Teff"])
        logg = (tic_params["logg"], tic_params["e_logg"])
        feh = (tic_params["MH"], tic_params["e_MH"])
    else:
        errmsg = "provide `tic_params`"
        assert tic_params is not None, errmsg

    coeffs = ld.claret(
        band=band,
        teff=teff[0],
        uteff=teff[1],
        logg=logg[0],
        ulogg=logg[1],
        feh=feh[0],
        ufeh=feh[1] ** kwargs,
    )
    return coeffs


def compute_cdpp(time, flux, window, cadence=0.5, robust=False):
    """
    Compute the CDPP in a given time window.
    See https://github.com/dfm/ketu/blob/master/ketu/cdpp.py

    :param time:
        The timestamps measured in days.
    :param flux:
        The time series. This should either be the raw data or normalized to
        unit mean (not relative flux with zero mean).
    :param window:
        The window in hours.
    :param cadence: (optional)
        The cadence of the observations measured in hours.
    :param robust: (optional)
        Use medians instead of means.
    :returns cdpp:
        The computed CDPP in ppm.
    """
    # Mask missing data and fail if no acceptable points exist.
    m = np.isfinite(time) * np.isfinite(flux)
    if not np.sum(m):
        return np.inf
    t, f = time[m], flux[m]

    # Compute the running relative std deviation.
    std = np.empty(len(t))
    hwindow = 0.5 * window
    for i, t0 in enumerate(t):
        m = np.abs(t - t0) < hwindow
        if np.sum(m) <= 0:
            std[i] = np.inf
        if robust:
            mu = np.median(f[m])
            std[i] = np.sqrt(np.median((f[m] - mu) ** 2)) / mu
        else:
            std[i] = np.std(f[m]) / np.mean(f[m])

    # Normalize by the window size.
    return 1e6 * np.median(std) / np.sqrt(window / cadence)


def get_pix_area_threshold(Tmag):
    """get pixel area based on Tmag, max=13 pix
    Taken from vineyard/vinify.py
    """
    # set a threshold for the number of pixel by Tmag
    area_len = 9 - np.fix(Tmag / 2)
    # 最大値を7*7に制限
    # restrict the maximam as 7*7
    area_len = min(area_len, 7)
    # 最小値を3*3に制限
    # restrict the minimum as 3*3
    area_len = max(area_len, 3)
    return area_len ** 2


# def determine_aperture(img, center, area_thresh=9):
#     """determine aperture
#     Taken from vineyard/aperture.py
#     """
#     mid_val = np.nanmedian(img)
#     img = np.nan_to_num(img)
#     #統計量を求める。
#     # calculate statics
#     flat_img = np.ravel(img)
#     Q1 = stats.scoreatpercentile(flat_img, 25)
#     Q3 = stats.scoreatpercentile(flat_img, 75)
#     Q_std = Q3 - Q1
#     #星中心を算出
#     # calculate the center of the star
#     center_tuple = tuple(np.round(center).astype(np.uint8))
#     #3Qstd以上の切り出し領域を求める
#     # calculate the cut area whose flux is larger than 3 Qstd
#     contours = trim_aperture(img, 3, mid_val, Q_std, area_thresh)
#     #4Qstd以上の切り出し領域を求める
#     # calculate the cut area whose flux is larger than 4 Qstd
#     contours.extend(trim_aperture(img, 4, mid_val, Q_std, area_thresh))
#     for contour in contours:
#         #中心が含まれているか確認
#         # check whether the contour contains the central pixel
#         has_center = cv2.pointPolygonTest(contour, center_tuple, False)
#         if has_center >= 0:
#             #apertureを作成
#             # make aperture
#             aperture = np.zeros_like(img).astype(np.uint8)
#             cv2.fillConvexPoly(aperture, points=contour, color=1)
#             #近傍星がないか確認
#             # check whether the aperture is contaminated
#             if not has_nearby_star(img, aperture):
#                 break
#     #決めかねてしまう場合
#     # if aperture cannot be determined by above process
#     else:
#         #中心含む4pixをapertureにする
#         # aperture is nearest 4 pixels from the center of the star
#         offset = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]])
#         aperture_contour = np.round(center + offset).astype(np.int32)
#         aperture = np.zeros_like(img).astype(np.uint8)
#         cv2.fillConvexPoly(aperture, points=aperture_contour, color=1)
#     return aperture
def get_RV_K(
    P_days,
    mp_Mearth,
    Ms_Msun,
    ecc=0.0,
    inc_deg=90.0,
    nsamples=10000,
    percs=[50, 16, 84],
    return_samples=False,
    plot=False,
):
    """Compute the RV semiamplitude in m/s via Monte Carlo
    P_days : tuple
        median and 1-sigma error
    mp_Mearth : tuple
        median and 1-sigma error
    Ms_Msun : tuple
        median and 1-sigma error
    """
    if np.all(
        isinstance(P_days, tuple),
        isinstance(Ms_Msun, tuple),
        isinstance(mp_Mearth, tuple),
    ):
        # generate samples
        P_days = np.random.rand(nsamples) * P_days[1] + P_days[0]
        mp_Mearth = np.random.rand(nsamples) * mp_Mearth[1] + mp_Mearth[0]
        Ms_Msun = np.random.rand(nsamples) * Ms_Msun[1] + Ms_Msun[0]
    P = P_days * u.day.to(u.second) * u.second
    Ms = Ms_Msun * u.Msun.to(u.kg) * u.kg
    mp = mp_Mearth * u.Mearth.to(u.kg) * u.kg
    inc = np.deg2rad(inc_deg)
    K_samples = (
        (2 * np.pi * c.G / (P * Ms * Ms)) ** (1.0 / 3)
        * mp
        * np.sin(inc)
        / unumpy.sqrt(1 - ecc ** 2)
    ).value
    K, K_lo, K_hi = np.percentile(K_samples, percs)
    K, K_siglo, K_sighi = K, K - K_lo, K_hi - K
    if plot:
        _ = hist(K_samples, bins="scott")
    if return_samples:
        return (K, K_siglo, K_sighi, K_samples)
    else:
        return (K, K_siglo, K_sighi)


def get_RM_K(vsini_kms, rp_Rearth, Rs_Rsun):
    """Compute the approximate semi-amplitude for the Rossiter-McLaughlin
    effect in m/s given sky-projected rotation velocity and depth"""
    D = (rp_Rearth * u.Rearth.to(u.m) / Rs_Rsun * u.Rsun.to(u.m)) ** 2
    return (vsini_kms * D / (1 - D)) * 1e3


def get_above_lower_limit(lower, data_mu, data_sig, sigma=1):
    idx = norm.cdf(lower, loc=data_mu, scale=data_sig) < norm.cdf(sigma)
    return idx


def get_below_upper_limit(upper, data_mu, data_sig, sigma=1):
    idx = norm.cdf(upper, loc=data_mu, scale=data_sig) > norm.cdf(-sigma)
    return idx


def get_between_limits(lower, upper, data_mu, data_sig, sigma=1):
    idx = get_above_lower_limit(
        lower, data_mu, data_sig, sigma=sigma
    ) & get_below_upper_limit(upper, data_mu, data_sig, sigma=sigma)
    return idx


def map_float(x):
    return list(map(float, x))


def map_int(x):
    return list(map(int, x))


def reduce_list(l):
    rl = np.unique(reduce(concat, l))
    return rl


def split_func(x):
    return x.replace(" ", "").replace("_", "").split(",")


def flatten_list(lol):
    """flatten list of list (lol)"""
    return list(itertools.chain.from_iterable(lol))


def get_secondary_eclipse_threshold(flat, t14, per, t0, factor=3):
    """
    get the mean of the std x sigma of binned out-of-eclipse lightcurve
    useful as a constraint in fpp.ini file in vespa.
    This effectively means no secondary eclipse is detected above this level.

    flat : lk.LightCurve
        flattened light curve where transits will be masked
    factor : float
        factor = 3 means 3-sigma
    """
    tmask = get_transit_mask(flat, period=per, epoch=t0, duration_hours=t14)
    fold = flat[~tmask].fold(period=per, t0=t0)

    means = []
    chunks = np.arange(-0.5, 0.51, t14 / 24 / per)
    for n, x in enumerate(chunks):
        if n == 0:
            x1 = -0.5
            x2 = x
        elif n == len(chunks):
            x1 = x
            x2 = 0.5
        else:
            x1 = chunks[n - 1]
            x2 = x
        idx = (fold.phase > x1) & (fold.phase < x2)
        mean = np.nanmean(fold.flux[idx])
        print(mean)
        means.append(mean)

    return factor * np.nanstd(means)

# -*- coding: utf-8 -*-

r"""
general helper functions
"""

# Import standard library
import logging
from glob import glob
from operator import concat
from functools import reduce

# Import from standard package
from os.path import join, exists
import os

# Import from module
# from matplotlib.figure import Figure
# from matplotlib.image import AxesImage
# from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import lightkurve as lk
from astropy import units as u
from astropy.timeseries import LombScargle
from astropy.modeling import models, fitting
from astropy.io import ascii
from scipy.ndimage import zoom

# from astropy.io import ascii
from astropy.coordinates import (
    SkyCoord,
    Distance,
    sky_coordinate,
    Galactocentric,
    match_coordinates_3d,
)
from skimage import measure
from astroquery.mast import Catalogs
from astroquery.gaia import Gaia
from tqdm import tqdm
import deepdish as dd

# Import from package
from chronos import target, cluster
from chronos.config import DATA_PATH

log = logging.getLogger(__name__)

__all__ = [
    "get_tois",
    "get_toi",
    "get_target_coord",
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
    "get_harps_RV",
    "get_specs_table_from_tfop",
    "get_rotation_period",
    "get_transit_mask",
    "get_mag_err_from_flux",
    "get_err_quadrature",
    "map_float",
    "map_int",
    "detrend",
    "query_tpf",
    "query_tpf_tesscut",
    "is_gaiaid_in_cluster",
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


def is_gaiaid_in_cluster(gaiaid, cluster_name, catalog_name="Bouma2019"):
    # reduce the redundant names above
    c = cluster.Cluster(
        catalog_name=catalog_name, cluster_name=cluster_name, verbose=False
    )
    df_mem = c.query_cluster_members()
    if df_mem.source_id.isin([gaiaid]).sum() > 0:
        return True
    else:
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


def detrend(self, break_tolerance=10):
    """mainly to be added as method to lk.LightCurve
    """
    lc = self.copy()
    half = lc.time.shape[0] // 2
    if half % 2 == 0:
        # add 1 if even
        half += 1
    return lc.flatten(
        window_length=half, polyorder=1, break_tolerance=break_tolerance
    )


def get_rotation_period(
    time, flux, min_per=0.5, max_per=None, npoints=20, plot=True, verbose=True
):
    """
    time, flux : array
        time and flux
    max_period : float
        maxmimum period (default=half baseline e.g. ~13 days)
    npoints : int
        datapoints around which to fit a Gaussian

    The period and uncertainty were determined from the mean and the
    half-width at half-maximum of a Gaussian fit to the periodogram peak, respectively
    """
    baseline = int(time[-1] - time[0])
    max_per = max_per if max_per is not None else baseline / 2

    ls = LombScargle(time, flux)
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


def get_transit_mask(lc, period, epoch, duration_hours):
    """
    lc : lk.LightCurve
        lightcurve that contains time and flux properties
    """
    assert isinstance(lc, lk.LightCurve)
    temp_fold = lc.fold(period, t0=epoch)
    fractional_duration = (duration_hours / 24.0) / period
    phase_mask = np.abs(temp_fold.phase) < (fractional_duration * 1.5)
    transit_mask = np.in1d(lc.time, temp_fold.time_original[phase_mask])
    return transit_mask


def get_harps_RV(target_coord, separation=30, outdir=DATA_PATH, verbose=True):
    """
    Check if target has archival HARPS data from:
    http://www.mpia.de/homes/trifonov/HARPS_RVBank.html
    """
    fp = os.path.join(outdir, "HARPS_RVBank_table.csv")
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        msg = f"Loaded: {fp}\n"
    else:
        if verbose:
            print("This may take a while...")
        # csvurl = "http://www.mpia.de/homes/trifonov/HARPS_RVBank_v1.csv"
        # df = pd.read_csv(csvurl)
        homeurl = "http://www.mpia.de/homes/trifonov/HARPS_RVBank.html"
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
            print(f"There are {len(res)} matches: {res['Target'].values}")
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
            f"Nearest HARPS obj to target is\n{nearest_obj}: ra,dec=({ra},{dec})\n"
        )
        print(f'Try angular distance larger than d={sep2d.arcsec:.4f}"\n')
        return None


def get_mamajek_table(clobber=False, verbose=True, data_loc=DATA_PATH):
    fp = join(data_loc, f"mamajek_table.csv")
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


def get_err_quadrature(err1, err2):
    return np.sqrt(err1 ** 2 + err2 ** 2)


def get_absolute_gmag(gmag, distance, a_g):
    """
    gmag : float
        apparent G band magnitude
    distance : float
        distance in pc
    a_g : float
        extinction in the G-band
    """
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
    # ratio of A_X/A_V
    if color == "bp_rp":
        # E(Bp-Rp) = A_Bp-A_Rp
        Ag_Av = extinction_ratios["G"]
        Ab_Av = extinction_ratios["Bp"]
        Ar_Av = extinction_ratios["Rp"]
        Ab_minus_Ar = (A_g / Ag_Av) * (Ab_Av - Ar_Av)  # difference
    else:
        raise NotImplementedError
    return Ab_minus_Ar


def get_absolute_color_index(A_g, bmag, rmag):
    """
    Deredden the Gaia Bp-Rp color using Bp-Rp extinction ratio (==Bp-Rp excess)

    E(Bp-Rp) = A_Bp - A_Rp = (Bp-Rp)_obs - (Bp-Rp)_abs
    --> (Bp-Rp)_abs = (Bp-Rp)_obs - E(Bp-Rp)

    Note that 'bmag-rmag' is same as bp_rp column in gaia table
    See also http://www.astro.ncu.edu.tw/~wchen/Courses/ISM/11.Extinction.pdf
    """
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
    if sap_mask == "all":
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
        mask = tpf.pipeline_mask  # default
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
    h, w = img.shape
    if xy_center is None:  # use the middle of the image
        y, x = np.unravel_index(np.argmax(img), img.shape)
        xy_center = [x, y]
        # check if near edge
        if np.any([x >= h - 1, x >= w - 1, y >= h - 1, y >= w - 1]):
            print("Brightest star is detected near the edges.")
            print("Aperture mask is placed at the center instead.\n")
            xy_center = [img.shape[0] // 2, img.shape[1] // 2]

    Y, X = np.ogrid[:h, :w]
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
    h = w = size
    if xy_center is None:  # use the middle of the image
        y, x = np.unravel_index(np.argmax(img), img.shape)
        xy_center = [x, y]
        # check if near edge
        if np.any([x >= h - 1, x >= w - 1, y >= h - 1, y >= w - 1]):
            print(
                "Brightest star detected is near the edges.\nAperture mask is placed at the center instead.\n"
            )
            x, y = img.shape[0] // 2, img.shape[1] // 2
            xy_center = [x, y]
    mask = np.zeros_like(img, dtype=bool)
    mask[y - h : y + h + 1, x - w : x + w + 1] = True
    # if angle:
    #    #rotate mask
    #    mask = rotate(mask, angle, axes=(1, 0), reshape=True, output=bool, order=0)
    return mask


def remove_bad_data(tpf, sector=None, verbose=True):
    """Remove bad cadences identified in data releae notes

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
        # use of Cam1 in attitude control was disabled at the
        # start of both orbits due to strong scattered light
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
        # use of Cam1 in attitude control was disabled at the
        # start of both orbits due to strong scattered light
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
        # use of Cam1 in attitude control was disabled at the
        # start of both orbits due to strong scattered light
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
    return tpf


def get_tois(
    clobber=True,
    outdir=DATA_PATH,
    verbose=False,
    remove_FP=True,
    remove_known_planets=False,
    add_FPP=True,
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
    else:
        d = pd.read_csv(fp).drop_duplicates()
        msg = f"Loaded: {fp}\n"
    d.to_csv(fp, index=False)

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


def get_toi(toi, clobber=True, outdir=DATA_PATH, add_FPP=False, verbose=True):
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

    df = get_tois(clobber=clobber, verbose=verbose, outdir=outdir)

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
    ra=None, dec=None, toi=None, tic=None, epic=None, gaiaid=None, name=None
):
    """get target coordinate
    """

    if np.all([ra, dec]):
        target_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    # TIC
    elif toi:
        toi_params = get_toi(toi=toi, clobber=False, verbose=False)
        target_coord = SkyCoord(
            ra=toi_params["RA"].values[0],
            dec=toi_params["Dec"].values[0],
            distance=toi_params["Stellar Distance (pc)"].values[0],
            unit=(u.hourangle, u.degree, u.pc),
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
        star = client.k2_star(int(epic))
        ra = float(star.k2_ra)
        dec = float(star.k2_dec)
        target_coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        # target_coord = SkyCoord.from_name(f"EPIC {epic}")
    elif gaiaid is not None:
        target_coord = SkyCoord.from_name(f"Gaia DR2 {gaiaid}")
    elif name is not None:
        target_coord = SkyCoord.from_name(name)
    else:
        raise ValueError("Supply RA & Dec, TOI, TIC, or Name")
    return target_coord


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
                t = target.Target(toiid=toi, verbose=verbose)
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
            print(f"Querying Gaia DR2 catalog for new TOIs\n")
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


# def get_K2_targetlist(campaign, verbose=True):
#     """
#     campaign: K2 campaign number [0-18]
#     """
#     if verbose:
#         print("Retrieving K2 campaign {} target list...\n".format(campaign))
#
#     outdir = "../data/K2targetlist/"
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
#
#
# def get_DR2_cluster_members(clustername, verbose=True):
#     """
#     cluster (row, column)
#     Hyades (515, 7)
#     IC2391 (327, 7)
#     IC2602 (494, 7)
#     Blanco1 (489, 7)
#     ComaBer (153, 7)
#     NGC2451 (404, 7)
#     Pleiades (1332, 7)
#     Praesepe (946, 7)
#     alphaPer (743, 7)
#     "all": get all cluster member
#     Table source: https://www.cosmos.esa.int/web/gaia/dr2-papers
#     """
#     cluster = clustername.strip().lower()
#     link = "../data/TablesGaiaDR2HRDpaper/TableA1a.csv"
#     cluster_members = pd.read_csv(link, delimiter=",")
#     # remove redundant spaces in column
#     cols = ["".join(col.split()) for col in cluster_members.columns]
#     cluster_members.columns = cols
#
#     clusters = {}
#     for c, df in cluster_members.groupby(by="Cluster"):
#         c = c.strip().lower()
#         cols = ["".join(col.split()) for col in df.columns]
#         clusters[c] = df
#
#     if cluster == "all":
#         if verbose:
#             print(
#                 "Retrieving {} known DR2 cluster members...\n".format(cluster)
#             )
#         # print(c,df.shape)
#         return clusters
#     else:
#         if verbose:
#             print("Retrieving {} cluster members...\n".format(cluster))
#         return clusters[cluster]
#
#
# def get_ra_dec_mag(epicnum, verbose=False):
#     if verbose:
#         print("\nquerying RA and DEC...\n")
#     epic = client.k2_star(int(epicnum))
#     ra = float(epic.k2_ra)
#     dec = float(epic.k2_dec)
#     mag = float(epic.kp)
#     return ra, dec, mag
#
#
# def get_cluster_centroid(clustername):
#     cluster = get_DR2_cluster_members(clustername)
#     ra = np.nanmedian(cluster["ra"])
#     dec = np.nanmedian(cluster["dec"])
#     return ra, dec


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


def get_limbdark(band, tic_params, teff=None, logg=None, feh=None, **kwargs):
    """
    """
    try:
        import limbdark as ld
    except Exception:
        command = (
            "pip install git+https://github.com/john-livingston/limbdark.git"
        )
        raise ModuleNotFoundError(command)

    coeffs = ld.claret(
        band=band,
        teff=teff[0] if np.isnan(tic_params["Teff"]) else tic_params["Teff"],
        uteff=teff[1]
        if np.isnan(tic_params["e_Teff"])
        else tic_params["e_Teff"],
        logg=logg[0] if np.isnan(tic_params["logg"]) else tic_params["logg"],
        ulogg=logg[1]
        if np.isnan(tic_params["e_logg"])
        else tic_params["e_logg"],
        feh=feh[0] if np.isnan(tic_params["MH"]) else tic_params["MH"],
        ufeh=feh[1] if np.isnan(tic_params["e_MH"]) else tic_params["e_MH"],
        **kwargs,
    )
    return coeffs


def map_float(x):
    return list(map(float, x))


def map_int(x):
    return list(map(int, x))


def reduce_list(l):
    rl = np.unique(reduce(concat, l))
    return rl


def split_func(x):
    return x.replace(" ", "").replace("_", "").split(",")

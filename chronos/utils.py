# -*- coding: utf-8 -*-

r"""
general helper functions
"""

# Import standard library
import logging
from glob import glob

# Import from standard package
from os.path import join, exists
import os

# Import from module
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from astropy import units as u
from astropy import constants as c
from astropy.io import ascii
from scipy.ndimage import zoom

# from astropy.io import ascii
from astropy.coordinates import (
    SkyCoord,
    Distance,
    Galactocentric,
    match_coordinates_3d,
)
from skimage import measure
from astroquery.mast import Catalogs
from astroquery.gaia import Gaia
from tqdm import tqdm
import deepdish as dd
import k2plr

# Import from package
from chronos import target
from chronos.config import DATA_PATH

client = k2plr.API()
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
    "parse_aperture_mask",
    "make_round_mask",
    "make_square_mask",
    "remove_bad_data",
    "is_point_inside_mask",
    "compute_fluxes_within_mask",
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


def check_harps_RV(target_coord, dist=30, verbose=True):
    """
    Check if target has archival HARPS data from:
    http://www.mpia.de/homes/trifonov/HARPS_RVBank.html
    """
    url = "http://www.mpia.de/homes/trifonov/HARPS_RVBank_v1.csv"
    table = pd.read_html(url, header=0)
    # choose first table
    df = table[0]
    # coordinates
    coords = SkyCoord(ra=df["RA"], dec=df["DEC"], unit=(u.hourangle, u.deg))
    # check which falls within `dist`
    idxs = target_coord.separation(coords) < dist
    if idxs.sum() > 0:
        # result may be multiple objects
        res = df[idxs]

        if verbose:
            msg = "There are {} matches: {}".format(
                len(res), res["Target"].values
            )
            print(msg)
            #             logging.info(msg)
            print("{}\n\n".format(df.loc[idxs, df.columns[7:14]].T))
        return res

    else:
        # find the nearest HARPS object in the database to target
        idx, sep2d, dist3d = match_coordinates_3d(
            target_coord, coords, nthneighbor=1
        )
        nearest_obj = df.iloc[[idx]]["Target"].values[0]
        ra, dec = df.iloc[[idx]][["RA_deg", "DEC_deg"]].values[0]
        msg = "Nearest HARPS obj to target is\n{}: ra,dec=({:.4f},{:.4f})\n".format(
            nearest_obj, ra, dec
        )
        print(msg)
        #         logging.info(msg)
        print(
            'Try angular distance larger than d={:.4f}"\n'.format(
                sep2d.arcsec[0]
            )
        )
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
            fp = join(outdir, "Giacalone2020/tab4.txt")
            classified = ascii.read(fp).to_pandas()
            fp = join(outdir, "Giacalone2020/tab5.txt")
            unclassified = ascii.read(fp).to_pandas()
            fpp = pd.concat(
                [
                    classified[["TOI", "FPP-2m", "FPP-30m"]],
                    unclassified[["TOI", "FPP"]],
                ],
                sort=True,
            )
            d = pd.merge(d, fpp, how="outer")
    else:
        d = pd.read_csv(fp)
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


def get_toi(toi, clobber=True, outdir=DATA_PATH, verbose=True):
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
        df = df[df["parallax"] > 0]
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


def compute_fluxes_within_mask(tpf, mask, gaia_sources):
    """compute relative fluxes of gaia sources within aperture
    """
    ra, dec = gaia_sources[["ra", "dec"]].values.T
    pix_coords = tpf.wcs.all_world2pix(np.c_[ra, dec], 0)
    contour_points = measure.find_contours(mask, level=0.1)[0]
    isinside = [
        is_point_inside_mask(contour_points, pix) for pix in pix_coords
    ]
    min_gmag = gaia_sources.loc[isinside, "phot_g_mean_mag"].min()
    fluxes = gaia_sources.loc[isinside, "phot_g_mean_mag"].apply(
        lambda x: 10 ** (0.4 * (min_gmag - x))
    )
    return fluxes

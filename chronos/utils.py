# -*- coding: utf-8 -*-

r"""
helper functions
"""

# Import standard library
import logging

# Import from standard package
from os.path import join, exists
import os

# Import from module
import numpy as np
import pandas as pd
import lightkurve as lk
from astropy import units as u
from astropy.io import ascii
from astropy.coordinates import SkyCoord, Distance, Galactocentric
from astroquery.gaia import Gaia
from tqdm import tqdm
import deepdish as dd

# Import from package
from chronos import search
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
]

def get_mamajek_table(clobber=False, verbose=True, data_loc=DATA_PATH):
    fp = join(data_loc, f"mamajek_table.csv")
    if not exists(fp) or clobber:
        url = 'http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt'
        #cols="SpT Teff logT BCv Mv logL B-V Bt-Vt G-V U-B V-Rc V-Ic V-Ks J-H H-Ks Ks-W1 W1-W2 W1-W3 W1-W4 Msun logAge b-y M_J M_Ks Mbol i-z z-Y R_Rsun".split(' ')
        df = pd.read_csv(url, skiprows=21, skipfooter=524, delim_whitespace=True, engine='python')
        # tab = ascii.read(url, guess=None, data_start=0, data_end=124)
        # df = tab.to_pandas()
        #replace ... with NaN
        df = df.replace(['...','....','.....'], np.nan)
        #replace header
        #df.columns = cols
        #drop last duplicate column
        df = df.drop(df.columns[-1], axis=1)
        # df['#SpT_num'] = range(df.shape[0])
        # df['#SpT'] = df['#SpT'].astype('category')

        #remove the : type in M_J column
        df['M_J'] = df['M_J'].apply(lambda x: str(x).split(':')[0])
        #convert columns to float
        for col in df.columns:
            if col=='#SpT':
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(float)
            # if col=='SpT':
            #     df[col] = df[col].astype('categorical')
            # else:
            #     df[col] = df[col].astype(float)
        df.to_csv(fp, index=False)
        print(f'Saved: {fp}')
    else:
        df = pd.read_csv(fp)
        if verbose:
            print(f'Loaded: {fp}')
    return df

def get_distance(m, M, Av):
    """
    calculate distance [in pc] from extinction-corrected magnitude
    using the equation: 10**(0.2*(m-M+5-Av))

    see http://astronomy.swin.edu.au/cosmos/I/Interstellar+Reddening

    Parameters
    ---------
    m : apparent magnitude
    M : absolute magnitude
    Av : extinction (in V band)
    """
    distance = 10**(0.2*(m-M+5-Av))
    return distance

def get_tois(
    clobber=True,
    outdir=DATA_PATH,
    verbose=False,
    remove_FP=True,
    remove_known_planets=False,
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
    dl_link = (
        "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    )
    fp = join(outdir, "TOIs.csv")
    if not exists(outdir):
        os.makedirs(outdir)

    if not exists(fp) or clobber:
        d = pd.read_csv(dl_link)  # , dtype={'RA': float, 'Dec': float})
        # remove False Positives
        if remove_FP:
            d = d[d["TFOPWG Disposition"] != "FP"]
            if verbose:
                print("TOIs with TFPWG disposition==FP are removed.\n")
        if remove_known_planets:
            planet_keys = [
                "WASP",
                "SWASP",
                "HAT",
                "HATS",
                "KELT",
                "QATAR",
                "K2",
                "Kepler",
            ]
            keys = []
            for key in planet_keys:
                idx = ~np.array(d["Comments"].str.contains(key).tolist(), dtype=bool)
                d = d[idx]
                if idx.sum() > 0:
                    keys.append(key)
            if verbose:
                print(f"{keys} planets are removed.\n")
        d.to_csv(fp, index=False)
        msg = f"Saved: {fp}\n"
    else:
        d = pd.read_csv(fp)
        # remove False Positives
        if remove_FP:
            d = d[d["TFOPWG Disposition"] != "FP"]
        msg = f"Loaded: {fp}"
    if verbose:
        print(msg)

    return d.sort_values("TOI")


def get_toi(toi=None, tic=None, clobber=True, outdir=DATA_PATH, verbose=True):
    """Query TOI from TOI list

    Parameters
    ----------
    tic : int
        TIC id
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

    if (toi is None) and (tic is None):
        raise ValueError("Provide toi or tic")
    else:
        if toi is not None:
            if isinstance(toi, int):
                toi = float(str(toi) + ".01")
            else:
                planet = str(toi).split(".")[1]
                assert len(planet) == 2, "use pattern: TOI.01"
            idx = df["TOI"].isin([toi])
        elif tic:
            idx = df["TIC ID"].isin([tic])
        q = df.loc[idx]
        # return if empty, else continue
        if len(q) == 0:
            raise ValueError("TOI not found!")

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
    elif np.any([toi, tic]):
        toi_params = get_toi(toi=toi, tic=tic, clobber=False, verbose=False)
        target_coord = SkyCoord(
            ra=toi_params["RA"].values[0],
            dec=toi_params["Dec"].values[0],
            distance=toi_params["Stellar Distance (pc)"].values[0],
            unit=(u.hourangle, u.degree, u.pc),
        )
    # name resolver
    elif epic is not None:
        target_coord = SkyCoord.from_name(f"EPIC {epic}")
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
    target_dist = Distance(parallax=star["parallax"] * u.mas)
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


def compare_pdc_sap(toiid):
    toi = get_toi(toi=toiid, verbose=False)
    period = toi["Period (days)"].values[0]
    t0 = toi["Epoch (BJD)"].values[0]
    tic = toi["TIC ID"].values[0]

    lcf = lk.search_lightcurvefile(f"TIC {tic}", mission="TESS").download()
    if lcf is not None:
        sap = lcf.SAP_FLUX.normalize()
        pdcsap = lcf.PDCSAP_FLUX.normalize()

        ax = sap.bin(11).fold(period=period, t0=t0).scatter(label="SAP")
        ax = pdcsap.bin(11).fold(period=period, t0=t0).scatter(ax=ax, label="PDCSAP")
        # ax.set_xlim(-0.1,0.1)
        ax.set_title(f"TOI {toiid}")
    return lcf, ax


def query_gaia_params_of_all_tois(fp=None, verbose=True, clobber=False, update=True):
    """
    """
    if fp is None:
        fp = join(DATA_PATH, "toi_gaia_params.hdf5")

    tois = get_tois(verbose=verbose)
    toiids = np.unique(tois.TOI.astype(float))
    if not exists(fp) or clobber:
        # download all from gaia catalog
        if verbose:
            print(f"Querying Gaia DR2 catalog for {len(toiids)} TOIs\n")
        toi_gaia_params = {}
        for toi in tqdm(toiids):
            try:
                t = search.Target(toiid=toi, verbose=verbose)
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
                    t = search.Target(toiid=toi, verbose=verbose)
                    # query gaia dr2 catalog to get gaia id
                    df_gaia = t.query_gaia_dr2_catalog(return_nearest_xmatch=True)
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

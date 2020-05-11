# -*- coding: utf-8 -*-
"""
test methods of target module
* target identification
* target coordinate resolution
* catalog querries: gaia, tic, mast, vizier, simbad
* gaia and tic cross-matching
* cluster catalog cross-matching
"""
import numpy as np
import pandas as pd
from astropy.table import Table
from astroquery.utils import TableList
from chronos.target import Target


def test_toi_tic():
    # toi
    t1 = Target(toiid=837)
    assert t1.toiid is not None
    coord1 = t1.target_coord
    assert coord1 is not None
    # tic
    t2 = Target(ticid=460205581)
    assert t2.ticid is not None
    coord2 = t2.target_coord
    assert coord2 is not None
    assert np.allclose(
        [coord1.ra.deg, coord1.dec.deg],
        [coord2.ra.deg, coord2.dec.deg],
        rtol=1e-2,
    )


def test_name():
    # name
    t = Target(name="Pi Mensae")
    assert t.target_name is not None
    coord = t.target_coord
    assert coord is not None


def test_epic_name():
    # epic
    t1 = Target(epicid=201270176)
    assert t1.epicid is not None
    coord1 = t1.target_coord
    assert coord1 is not None

    t2 = Target(name="EPIC 201270176")
    assert t2.target_name is not None
    # assert t2.epicid is not None
    coord2 = t2.target_coord
    assert coord2 is not None
    assert np.allclose(
        [coord1.ra.deg, coord1.dec.deg],
        [coord2.ra.deg, coord2.dec.deg],
        rtol=1e-2,
    )


# def test_k2():
#     # k2
#     t = Target(name="K2-33")
#     coord = t.target_coord
#     assert (coord is not None) & (coord is not None)


def test_radec():
    # ra,dec
    t = Target(ra_deg=157.03729167, dec_deg=-64.50521111)
    coord = t.target_coord
    assert coord is not None


def test_gaia():
    # gaia
    t = Target(gaiaDR2id=5251470948229949568)
    assert t.gaiaid is not None
    coord = t.target_coord
    assert coord is not None


def test_query_gaia():
    t = Target(gaiaDR2id=5251470948229949568)
    gaia_params1 = t.query_gaia_dr2_catalog(radius=60)
    assert isinstance(gaia_params1, pd.DataFrame)
    gaia_params2 = t.query_gaia_dr2_catalog(return_nearest_xmatch=True)
    assert isinstance(gaia_params2, pd.Series)


def test_query_tic():
    t = Target(ticid=460205581)
    tic_params1 = t.query_tic_catalog(radius=60)
    assert isinstance(tic_params1, pd.DataFrame)
    tic_params2 = t.query_tic_catalog(return_nearest_xmatch=True)
    assert isinstance(tic_params2, pd.Series)


def test_target_xmatch():
    t = Target(toiid=837)
    _ = t.query_gaia_dr2_catalog(return_nearest_xmatch=True)
    _ = t.query_tic_catalog(return_nearest_xmatch=True)
    assert t.validate_gaia_tic_xmatch()


def test_tic_match():
    """test tic match using query_tic_catalog & native vizier search
    TIC v8: https://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=IV/38/tic
    """
    t = Target(toiid=837)
    tab = t.query_vizier()
    df = tab["IV/38/tic"].to_pandas()
    d1 = df.iloc[0]
    d2 = t.query_tic_catalog(return_nearest_xmatch=True)
    assert int(d1["TIC"]) == int(d2["ID"])


def test_harps():
    t = Target(toiid=200)
    df = t.query_harps_bank_table()
    assert isinstance(df, pd.DataFrame)


def test_eso():
    t = Target(ticid=410214986)
    df = t.query_eso()
    assert isinstance(df, pd.DataFrame)


def test_specs_tfop():
    t = Target(ticid=410214986)
    df = t.query_specs_from_tfop()
    assert isinstance(df, pd.DataFrame)


def test_mast():
    t = Target(ticid=410214986)
    df = t.query_mast()
    assert isinstance(df, pd.DataFrame)


def test_simbad():
    t = Target(ticid=410214986)
    df = t.query_simbad()
    assert isinstance(df, pd.DataFrame)


def test_vizier():
    t = Target(ticid=410214986, verbose=False)
    tables = t.query_vizier()
    assert isinstance(tables, TableList)
    assert isinstance(tables[0], Table)


def test_find_cluster():
    t = Target(toiid=837)
    _ = t.query_gaia_dr2_catalog(return_nearest_xmatch=True)
    nearest_cluster_member = t.get_nearest_cluster_member(
        catalog_name="CantatGaudin2020", match_id=True
    )
    assert isinstance(nearest_cluster_member, pd.Series)
    assert t.nearest_cluster_name == "IC_2602"

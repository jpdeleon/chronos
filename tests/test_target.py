# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from package
from chronos.target import Target


def test_target_coord():
    # name
    target_coord = Target(name="Pi Mensae").target_coord
    assert np.any([target_coord.ra, target_coord.dec])
    # k2
    target_coord = Target(name="K2-33").target_coord
    assert np.any([target_coord.ra, target_coord.dec])
    # toi
    t1 = Target(toiid=837)
    assert len(t1.toiid) is not None
    assert np.all([t1.target_coord.ra, t1.target_coord.dec])
    # tic
    t2 = Target(ticid=460205581)
    assert len(t2.ticid) is not None
    assert np.all([t2.ra, t2.dec])
    assert t1.ra == t2.ra
    assert t1.dec == t2.dec
    # ra,dec
    target_coord = Target(
        ra_deg=157.03729167, dec_deg=-64.50521111
    ).target_coord
    assert np.all([target_coord.ra, target_coord.dec])
    # gaia
    t = Target(gaiaDR2id=5251470948229949568)
    assert len(t.gaiaid) is not None
    assert np.all([t.target_coord.ra, t.target_coord.dec])
    # epic
    t1 = Target(epicid=201270176)
    assert len(t1.epicid) > 0
    assert np.all([t1.target_coord.ra, t1.target_coord.dec])
    t2 = Target(name="EPIC 201270176")
    assert np.any([t2.target_coord.ra, t2.target_coord.dec])
    assert len(t2.epicid) > 0
    assert t1.target_coord.ra == t2.target_coord.ra
    assert t1.target_coord.dec == t2.target_coord.dec


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

# -*- coding: utf-8 -*-
import pandas as pd
import lightkurve as lk
from chronos.k2 import K2, Everest, K2sff

EPICID = 211916756  # k2-95
CAMPAIGN = 5  # or 18


def test_k2_attributes():
    """
    """
    # test inherited attributes
    s = K2(epicid=EPICID, campaign=CAMPAIGN)
    assert s.epicid is not None
    assert s.target_coord is not None
    gaia_params = s.query_gaia_dr2_catalog(return_nearest_xmatch=True)
    assert isinstance(gaia_params, pd.Series)
    tic_params = s.query_tic_catalog(return_nearest_xmatch=True)
    assert isinstance(tic_params, pd.Series)


def test_k2_lc_pipeline():
    s = K2(epicid=EPICID, campaign=CAMPAIGN)
    s.get_lc("sap")
    assert isinstance(s.lc_sap, lk.LightCurve)
    s.get_lc("pdcsap")
    assert isinstance(s.lc_pdcsap, lk.LightCurve)


# def test_k2_lc_custom():
#     s = K2(epicid=EPICID, campaign=CAMPAIGN)
#     sap = s.make_custom_lc()


def test_k2_tpf():
    s = K2(epicid=EPICID, campaign=CAMPAIGN)
    tpf = s.get_tpf()
    assert isinstance(tpf, lk.targetpixelfile.TargetPixelFile)


def test_everest():
    """
    """
    s = Everest(epicid=EPICID, campaign=CAMPAIGN)
    assert isinstance(s.lc_everest, lk.LightCurve)


def test_k2sff():
    """
    """
    s = K2sff(epicid=EPICID, campaign=CAMPAIGN)
    assert isinstance(s.lc_k2sff, lk.LightCurve)

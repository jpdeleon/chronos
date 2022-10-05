# -*- coding: utf-8 -*-
"""
test methods of k2 module
"""
import pytest
import pandas as pd
import lightkurve as lk
from chronos.k2 import K2, Everest, K2sff, _KeplerLightCurve

# from matplotlib.figure import Figure
# from matplotlib.axes import Axes

EPICID = 211916756  # k2-95
CAMPAIGN = 5  # or 18

s = K2(epicid=EPICID, campaign=CAMPAIGN)


def test_k2_attributes():
    """ """
    # test inherited attributes
    assert s.epicid is not None
    assert s.target_coord is not None
    gaia_params = s.query_gaia_dr2_catalog(return_nearest_xmatch=True)
    assert isinstance(gaia_params, pd.Series)
    tic_params = s.query_tic_catalog(return_nearest_xmatch=True)
    assert isinstance(tic_params, pd.Series)


def test_k2_lc_pipeline():
    assert isinstance(s.lc_sap, lk.LightCurve)
    assert isinstance(s.lc_pdcsap, lk.LightCurve)


def test_k2_methods():
    lcs = K2.get_lightcurves(EPICID)
    # assert isinstance(lcs, lk.LightCurve)
    assert isinstance(lcs, _KeplerLightCurve)


@pytest.mark.skip
def test_k2_lc_custom():
    s = K2(epicid=EPICID, campaign=CAMPAIGN)
    sap = s.make_custom_lc()
    assert hasattr(sap, "plot")


def test_k2_tpf():
    tpf = s.get_tpf()
    assert isinstance(tpf, lk.targetpixelfile.TargetPixelFile)


def test_everest():
    s = Everest(epicid=EPICID, campaign=CAMPAIGN, verbose=False)
    assert isinstance(s.lc_everest, lk.LightCurve)


def test_k2sff():
    s = K2sff(epicid=EPICID, campaign=CAMPAIGN, verbose=False)
    assert isinstance(s.lc_k2sff, lk.LightCurve)


def test_k2_plots():
    ax = K2.plot_everest_k2sff_comparison(EPICID)
    assert hasattr(ax, "plot")

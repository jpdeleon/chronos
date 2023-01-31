# -*- coding: utf-8 -*-
"""
test methods of cdips module
"""
from matplotlib.axes import Axes
import lightkurve as lk
from chronos import TGLC

# TICID = 460205581
TOIID = 3353
GAIADR3ID = 5212899427468919296
SECTOR = 2
# QUALITY_BITMASK = "default"


p1 = TGLC(
    toiid=TOIID,
    sector=SECTOR,
    lctype="psf",
    # quality_bitmask=QUALITY_BITMASK,
)

p2 = TGLC(
    toiid=TOIID,
    sector=SECTOR,
    lctype="aperture",
    # quality_bitmask=QUALITY_BITMASK,
)


def test_tglc_init():
    assert isinstance(p1.lc, lk.LightCurve)
    assert isinstance(p2.lc, lk.LightCurve)


def test_tglc_plot():
    ax1 = p1.lc.plot()
    assert isinstance(ax1, Axes)
    ax2 = p2.lc.plot()
    assert isinstance(ax2, Axes)

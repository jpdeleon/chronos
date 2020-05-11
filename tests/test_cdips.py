# -*- coding: utf-8 -*-
"""
test methods of cdips module
"""
from matplotlib.axes import Axes
import lightkurve as lk
from chronos import CDIPS

TICID = 460205581
SECTOR = 10
QUALITY_BITMASK = "default"


cdips = CDIPS(
    ticid=TICID,
    sector=SECTOR,
    lctype="flux",
    aper_idx=1,
    quality_bitmask=QUALITY_BITMASK,
)


def test_cdips_init():
    assert isinstance(cdips.lc, lk.LightCurve)


def test_cdips_plot():
    ax = cdips.lc.plot()
    assert isinstance(ax, Axes)

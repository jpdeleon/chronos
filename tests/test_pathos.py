# -*- coding: utf-8 -*-
"""
test methods of cdips module
"""
from matplotlib.axes import Axes
import lightkurve as lk
from chronos import PATHOS

# TICID = 460205581
TOIID = 837
SECTOR = 10
QUALITY_BITMASK = "default"


p = PATHOS(
    # ticid=TICID,
    toiid=TOIID,
    sector=SECTOR,
    lctype="corr",
    aper_idx=1,
    quality_bitmask=QUALITY_BITMASK,
)


def test_cdips_init():
    assert isinstance(p.lc, lk.LightCurve)


def test_cdips_plot():
    ax = p.lc.plot()
    assert isinstance(ax, Axes)

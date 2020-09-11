# -*- coding: utf-8 -*-
"""
test methods of pathos module
"""
from matplotlib.axes import Axes
import lightkurve as lk
from chronos import PATHOS

# TICID = 460205581
TOIID = 837
SECTOR = 10
QUALITY_BITMASK = "default"


cdips = PATHOS(
    # ticid=TICID,
    toiid=TOIID,
    sector=SECTOR,
    lctype="flux",
    aper_idx=1,
    quality_bitmask=QUALITY_BITMASK,
)


def test_pathos_init():
    assert isinstance(cdips.lc, lk.LightCurve)


def test_cdips_plot():
    ax = cdips.lc.plot()
    assert isinstance(ax, Axes)

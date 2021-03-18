# -*- coding: utf-8 -*-
"""
test methods of cdips module
"""
from matplotlib.axes import Axes
import lightkurve as lk
from chronos import Diamante

TICID = 460205581
TOIID = 837
SECTOR = 10
QUALITY_BITMASK = "default"


d = Diamante(
    ticid=TICID,
    # toiid=TOIID,
    # sector=SECTOR,
    lc_num=1,
    aper_radius=2,
    # quality_bitmask=QUALITY_BITMASK,
)


def test_diamante_init():
    assert isinstance(d.lc, lk.LightCurve)


def test_diamante_plot():
    ax = d.lc.plot()
    assert isinstance(ax, Axes)

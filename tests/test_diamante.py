# -*- coding: utf-8 -*-
"""
test methods of cdips module

ex file:
https://archive.stsci.edu/hlsps/diamante/0000/0009/0167/4675/hlsp_diamante_tess_lightcurve_tic-0000000901674675_tess_v1_llc.fits
"""
from matplotlib.axes import Axes
import lightkurve as lk
from chronos import Diamante

TICID = 901674675
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

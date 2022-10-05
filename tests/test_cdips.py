# -*- coding: utf-8 -*-
"""
test methods of cdips module

ex. file:
https://archive.stsci.edu/hlsps/cdips/s0010/cam3_ccd1/hlsp_cdips_tess_ffi_gaiatwo0005228253248367065856-s0010-cam3-ccd1_tess_v01_llc.fits
"""
from matplotlib.axes import Axes
import lightkurve as lk
from chronos import CDIPS

# TICID = 460205581
TOIID = 837
SECTOR = 10
QUALITY_BITMASK = "default"


cdips = CDIPS(
    # ticid=TICID,
    toiid=TOIID,
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

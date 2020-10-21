# -*- coding: utf-8 -*-
"""
test methods of plot module
"""
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import lightkurve as lk
from chronos import Target, plot_archival_images

# TICID = 460205581
TOIID = 837
SECTOR = 10
QUALITY_BITMASK = "default"


t = Target(
    # ticid=TICID,
    toiid=TOIID,
    sector=SECTOR,
    quality_bitmask=QUALITY_BITMASK,
)


def test_archival():
    fig = plot_archival_images(
        ra=t.target_coord.ra.deg, dec=t.target_coord.dec.deg
    )
    assert isinstance(fig, Figure)

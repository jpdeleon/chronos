# -*- coding: utf-8 -*-
import lightkurve as lk
from chronos import CDIPS

TICID = 460205581
SECTOR = 10
QUALITY_BITMASK = "default"


def test_cdips():
    """
    """
    cdips = CDIPS(
        ticid=TICID,
        sector=SECTOR,
        lctype="flux",
        aper_idx=1,
        quality_bitmask=QUALITY_BITMASK,
    )
    assert isinstance(cdips.lc, lk.LightCurve)

# -*- coding: utf-8 -*-

# Import modules
import pytest
import lightkurve as lk

# Import from package
from chronos.tpf import Tpf, Tpf_cutout


def test_tpf():
    """
    confirm
    """
    ticid = 460205581
    sector = 10
    t = Tpf(
        ticid=ticid,
        sector=sector,
        quality_bitmask="default",
        apply_data_quality_mask=False,
    )
    tpf1 = t.get_tpf()

    res = lk.search_targetpixelfile(
        f"TIC {ticid}", mission="TESS", cadence="short"
    )
    tpf2 = res.download(sector=sector)
    assert tpf1.targetid == tpf2.targetid
    assert tpf1.sector == tpf2.sector
    assert tpf1.quality_bitmask == tpf2.quality_bitmask


def test_tpf_cutout():
    """
    confirm
    """
    ticid = 460205581
    sector = 10
    cutout_size = (15, 15)
    t = Tpf_cutout(
        ticid=ticid,
        sector=sector,
        quality_bitmask="default",
        apply_data_quality_mask=False,
        cutout_size=cutout_size,
    )
    tpf1 = t.get_tpf_tesscut()

    res = lk.search_tesscut(
        f"TIC {ticid}", mission="TESS", cutout_size=cutout_size
    )
    tpf2 = res.download(sector=sector)
    assert tpf1.targetid == tpf2.targetid
    assert tpf1.sector == tpf2.sector
    assert tpf1.quality_bitmask == tpf2.quality_bitmask

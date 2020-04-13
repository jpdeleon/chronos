# -*- coding: utf-8 -*-
import lightkurve as lk
from chronos.tpf import Tpf, FFI_cutout

TICID = 460205581
SECTOR = 10
CUTOUT_SIZE = (15, 15)


def test_tpf():
    """
    """
    t = Tpf(
        ticid=TICID,
        sector=SECTOR,
        quality_bitmask="default",
        apply_data_quality_mask=False,
    )
    tpf1 = t.get_tpf()
    assert tpf1.targetid == TICID
    assert tpf1.sector == SECTOR

    res = lk.search_targetpixelfile(
        f"TIC {t.ticid}", mission="TESS", cadence="short", sector=SECTOR
    )
    tpf2 = res.download()
    assert tpf1.targetid == tpf2.targetid
    assert tpf1.sector == tpf2.sector
    assert tpf1.quality_bitmask == tpf2.quality_bitmask


def test_tpf_cutout():
    """
    """
    t = FFI_cutout(
        ticid=TICID,
        sector=SECTOR,
        quality_bitmask="default",
        apply_data_quality_mask=False,
        cutout_size=CUTOUT_SIZE,
    )
    tpf1 = t.get_tpf_tesscut()
    assert tpf1.targetid == TICID
    assert tpf1.sector == SECTOR
    assert tpf1.flux.shape[1:] == CUTOUT_SIZE

    res = lk.search_tesscut(f"TIC {t.ticid}", sector=SECTOR)
    # assert tpf2.flux.shape[1:] == CUTOUT_SIZE

    tpf2 = res.download()
    assert f"TIC {tpf1.targetid}" == tpf2.targetid
    assert tpf1.sector == tpf2.sector
    assert tpf1.quality_bitmask == tpf2.quality_bitmask

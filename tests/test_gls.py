# -*- coding: utf-8 -*-
"""
test methods of gls module
"""
# from matplotlib.figure import Figure
from chronos import LongCadence, Gls

TICID = 460205581
SECTOR = 10


def test_gls():
    """
    """
    sc = LongCadence(ticid=TICID, sector=SECTOR)
    lc = sc.make_custom_lc()

    data = lc.time.value, lc.flux.value, lc.flux_err.value
    gls = Gls(data, Pbeg=1, verbose=False)
    assert isinstance(gls.best, dict)
    # fig = gls.plot(block=True)
    # fig = gls.plot(block=False)
    # assert isinstance(fig, Figure)

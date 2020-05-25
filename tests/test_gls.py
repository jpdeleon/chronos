# -*- coding: utf-8 -*-
from matplotlib.figure import Figure
from chronos import LongCadence, Gls

TICID = 460205581
SECTOR = 10


def test_gls():
    """
    """
    sc = LongCadence(ticid=TICID, sector=SECTOR)
    lc = sc.make_custom_lc()

    data = lc.time, lc.flux, lc.flux_err
    gls = Gls(data, Pbeg=1, verbose=False)
    fig = gls.plot(block=True)
    assert isinstance(fig, Figure)

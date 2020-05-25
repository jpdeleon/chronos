# -*- coding: utf-8 -*-
"""
test methods of planet module
* query harps bank rv data
"""
import numpy as np
from matplotlib.figure import Figure
from astroquery.utils import TableList
from chronos.planet import Planet

p = Planet(toiid=179)


def test_planet_init():
    tables = p.vizier_tables
    assert isinstance(tables, TableList)


def test_methods():
    Mp, Mp_siglo, Mp_sighi = p.get_Mp_from_MR_relation(use_toi_params=True)
    assert isinstance((Mp, Mp_siglo, Mp_sighi), tuple)
    Rp, Rp_siglo, Rp_sighi, Rp_samples = p.get_Rp_from_depth(
        return_samples=True
    )
    assert isinstance((Rp, Rp_siglo, Rp_sighi), tuple)
    assert isinstance(Rp_samples, np.ndarray)

    K, K_siglo, K_sighi, K_samples = p.get_RV_amplitude(return_samples=True)
    assert isinstance((K, K_siglo, K_sighi), tuple)
    assert isinstance(K_samples, np.ndarray)


def test_planet_plot():
    # _ = p.get_Rp_from_depth(plot=True)
    # assert isinstance(fig1, Figure)

    fig2 = p.plot_harps_rv_scatter()
    # fig3 = p.plot_harps_rv_gls()
    # p.plot_harps_rv_corr_matrix()
    assert isinstance(fig2, Figure)

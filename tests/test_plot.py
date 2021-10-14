# -*- coding: utf-8 -*-
"""
test methods of plot module
"""
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import lightkurve as lk
from chronos import ShortCadence
from chronos.plot import (
    plot_archival_images,
    get_dss_data,
    plot_dss_image,
    plot_possible_NEBs,
)

# TICID = 460205581
TOIID = 837


s = ShortCadence(
    # ticid=TICID,
    toiid=TOIID
)
lc = s.lc_pdcsap
gaia_sources = s.query_gaia_dr2_catalog(radius=60)


def test_archival():
    fig = plot_archival_images(
        ra=s.target_coord.ra.deg,
        dec=s.target_coord.dec.deg,
        survey1="dss1",
        survey2="poss2ukstu_red",  # or ps1 which requires panstarrs3 library
        return_baseline=False,
    )
    assert isinstance(fig, Figure)

    hdu = get_dss_data(ra=s.target_coord.ra.deg, dec=s.target_coord.dec.deg)
    ax = plot_dss_image(hdu)
    assert isinstance(ax, Axes)


# def test_Prot():
#     fig = plot_rotation_period(lc.time.value, lc.flux.value, lc.flux_err.value, npoints=5)
#     assert isinstance(fig, Figure)


def test_NEBs():
    ax = plot_possible_NEBs(gaia_sources, depth=0.001)
    assert isinstance(ax, Axes)


# def test_cluster_plot():
#     _ = plot_cluster_kinematics(toiid=TOIID)

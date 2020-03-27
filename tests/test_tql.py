# -*- coding: utf-8 -*-

# Import modules
import pytest

# Import from package
from chronos.utils import make_tql


def test_tql():
    make_tql(
        gaiaid=None,
        toiid=None,
        ticid=None,
        name=None,
        sector=None,
        cadence="long",
        sap_mask=None,
        aper_radius=1,
        threshold_sigma=5,
        percentile=90,
        cutout_size=(15, 15),
        quality_bitmask="default",
        apply_data_quality_mask=False,
        window_length=31,
        savefig=False,
        savetls=False,
        outdir=".",
        verbose=False,
        clobber=False,
    )

# -*- coding: utf-8 -*-

# Import modules
import pytest

# Import from package
from chronos.plot import make_tql

toiid = 1063
savefig = True
verbose = True
quality_bitmask = "default"
apply_data_quality_mask = False
cutout_size=(15, 15)
window_length = 0.5
lctype='custom'

if True:
    #name
    make_tql(
        gaiaid=None,
        toiid=None,
        ticid=460205581,
        name=None,
        sector=None,
        cadence="short",
        lctype="pdcsap",
        sap_mask="pipeline",
        cutout_size=cutout_size,
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )

if True:
    #square mask aper_radius=1
    make_tql(
        gaiaid=None,
        toiid=toiid,
        ticid=None,
        name=None,
        sector=None,
        cadence="short",
        lctype=lctype,
        sap_mask="square",
        aper_radius=1,
        cutout_size=cutout_size,
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )

if True:
    #square mask aper_radius=2
    make_tql(
        gaiaid=None,
        toiid=toiid,
        ticid=None,
        name=None,
        sector=None,
        cadence="short",
        lctype=lctype,
        sap_mask="square",
        aper_radius=2,
        cutout_size=cutout_size,
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )

if True:
    #round mask aper_radius=1
    make_tql(
        gaiaid=None,
        toiid=toiid,
        ticid=None,
        name=None,
        sector=None,
        cadence="short",
        lctype=lctype,
        sap_mask="round",
        aper_radius=1,
        cutout_size=cutout_size,
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )

if True:
    #round mask aper_radius=2, sector 11
    make_tql(
        gaiaid=None,
        toiid=toiid,
        ticid=None,
        name=None,
        sector=11,
        cadence="short",
        lctype=lctype,
        sap_mask="round",
        aper_radius=2,
        cutout_size=cutout_size,
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )

if True:
    #smaller cutout_size
    make_tql(
        gaiaid=None,
        toiid=toiid,
        ticid=None,
        name=None,
        sector=None,
        cadence="short",
        lctype=lctype,
        sap_mask="round",
        aper_radius=2,
        cutout_size=(10,10),
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )

if False:
    #name search
    make_tql(
        gaiaid=None,
        toiid=None,
        ticid=None,
        name="Trappist-1",
        sector=None,
        cadence="short",
        lctype=lctype,
        sap_mask="round",
        aper_radius=2,
        cutout_size=cutout_size,
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )

if True:
    #sap mask percentile
    make_tql(
        gaiaid=None,
        toiid=toiid,
        ticid=None,
        name=None,
        sector=None,
        cadence="short",
        lctype=lctype,
        sap_mask="percentile",
        percentile=90,
        cutout_size=cutout_size,
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )

if True:
    #sap mask threshold
    make_tql(
        toiid=toiid,
        cadence="short",
        lctype=lctype,
        sap_mask="threshold",
        threshold_sigma=5,
        cutout_size=cutout_size,
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )

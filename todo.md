# To do

## Basic
- [] query_gaia_dr2_catalog and tic catalog can be combined with galex into query_mast(catalog=["gaia","tic","galex"]) [see here](https://astroquery.readthedocs.io/en/latest/mast/mast.html) using [lower level functions here](https://mast.stsci.edu/api/v0/pyex.html)
- [] make input to cr.Target using get_target_coord unified as in mirai.parse_target_coord  
- [] make setup.py work in fresh conda environment
- [] remove redundant large files in /data; see [this repo](https://github.com/ideasrule/platon)
- [] make sure access to /data after installation works
- [] add tests following [this format](https://github.com/ljvmiranda921/seagull/blob/master/tests/test_board.py)
- [] configure documentation using [just-the-docs](https://github.com/pmarsceill/just-the-docs)
- [] standardize docstrings
- [] add binder, badges
- [] implement [typing](https://docs.python.org/3/library/typing.html) (> Python 3.8)
- [] include an environment.yml file, like [this](https://github.com/lgbouma/cdips_followup/blob/master/environment.yml)
- [x] check if caching works in get_lc(), get_tpf(), etc.
- [x] migrate constants to constants.py

## Functions
- [] incorporate (chronostar)[https://github.com/mikeireland/chronostar]
- [] mr-relation
- [] join multi-sector light curves
- [] See routines in [difference imaging analysis](https://github.com/ryanoelkers/DIA)
- [] add lightcurve from ASAS-SN project (Shappee et al. 2014; Kochanek et al. 2017))
- [] check [cdips-pipeline](https://github.com/waqasbhatti/cdips-pipeline)
- [] add [astrobase](https://github.com/waqasbhatti/astrobase) specifically the [astrotess module](https://astrobase.readthedocs.io/en/latest/astrobase.astrotess.html); see also [notebooks](https://github.com/waqasbhatti/astrobase-notebooks)
- [x] resolve make_custom_lc in LongCadence and ShortCadence (using tpf.py)
- [x] add detrend method; see https://stackoverflow.com/a/24865663/1910174
- [] check optimum [break_tolerance](https://github.com/KeplerGO/lightkurve/blob/master/lightkurve/lightcurve.py#L428) given cadence
- [] incorporate [triceratops](https://github.com/stevengiacalone/triceratops/tree/master/triceratops) in workflow
- [] fix HR diagram
- [] add isochrone fitting in CM diagram
- [] add size=stellar radius and color=logg from tic catalog in CMD/HRD similar to Hardegree-Ullman+2020 to show if dwarf/giant misclassification
- [] add data from [pathos pipeline](https://archive.stsci.edu/hlsp/pathos)
- [] See [Tess tutorials](https://outerspace.stsci.edu/display/TESS/6.0+-+Data+Search+Tutorials) and [notebooks](https://github.com/spacetelescope/notebooks)

## Notes
- [] see [TESS DR14 Notes S4.2](https://archive.stsci.edu/missions/tess/doc/tess_drn/tess_sector_14_drn19_v02.pdf) about transit search improvement in pipeline

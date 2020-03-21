# To do

## Basic
- [] check if caching works in get_lc(), get_tpf(), etc.
- [x] migrate constants to constants.py
- [] make setup.py work in fresh conda environment
- [] remove redundant large files in /data; see [this repo](https://github.com/ideasrule/platon)
- [] make sure access to /data after installation works
- [] add tests following [this format](https://github.com/ljvmiranda921/seagull/blob/master/tests/test_board.py)
- [] configure documentation using [just-the-docs](https://github.com/pmarsceill/just-the-docs)
- [] standardize docstrings
- [] add binder, badges
- [] implement [typing](https://docs.python.org/3/library/typing.html) (> Python 3.8)


## Functions
- [x] resolve make_custom_lc in LongCadence and ShortCadence (using tpf.py)
- [x] add detrend method; see https://stackoverflow.com/a/24865663/1910174
- [] check optimum [break_tolerance](https://github.com/KeplerGO/lightkurve/blob/master/lightkurve/lightcurve.py#L428) given cadence
- [] incorporate [triceratops](https://github.com/stevengiacalone/triceratops/tree/master/triceratops) in workflow
- [] fix HR diagram
- [] add isochrone fitting in CM diagram
- [] add size=stellar radius and color=logg from tic catalog in CMD/HRD similar to Hardegree-Ullman+2020 to show if dwarf/giant misclassification
- [] add lightcurve from ASAS-SN project (Shappee et al. 2014; Kochanek et al. 2017))

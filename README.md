# chronos
[![Build Status](https://travis-ci.com/jpdeleon/chronos.svg?branch=master)](https://travis-ci.com/jpdeleon/chronos)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


## Installation
```bash
$ git clone git@github.com:jpdeleon/chronos.git
$ cd chronos
$ pip install -e .
```


## test
```
$ pytest tests/
```


## Modules
* `target.py`: star bookkeeping, e.g. position, catalog cross-matching, archival data look-up
* `star.py`: physics-related calculations, e.g. extinction, spectral typing, isochrones, gyrochronology (inherits `target`)
* `planet.py`: planet parameters calculations (inherits `star`)
* `tpf.py`: targetpixel file manipulation
* `lightcurve.py`: light curve analysis either using SPOC or custom pipeline for short and long cadence (inherits `tpf`)
* `k2.py`: tpf and light curve for K2; likely to be ingested/refactored to tpf.py & lightcurve.py
* `cluster.py`: cluster catalog, cluster analysis + plotting
* `cdips.py` & `pathos.py`: api for CDIPS and PATHOS pipelines
* `plot.py`: custom plotting functionalities
* `utils.py`: useful utilities


## Dependencies
* `astropy` & `astroquery` for star and catalog bookkeeping
* `lightkurve`, `transitleastsquares`, & `wotan` for light curve analysis
* `emcee` & `corner` for MCMC analysis
* `isochrones` for isochrones analysis
* `dustmass` for extinction calculation
* `stardate` for gyrochronology
* [`mrexo`](https://github.com/shbhuk/mrexo) for mass-radius relation
* `triceratops` for FPP calculation based on lightc urve shape and contrast curve constraints

## For next update
* [`platon`](https://github.com/ideasrule/platon) for calculating transmission spectrum
* [`theJoker`](https://github.com/adrn/thejoker) for two-body MCMC sampler
* [`spock`](https://github.com/dtamayo/spock) for orbit stability analysis
* [`exoCMD`](https://github.com/gdransfield/ExoCMD)
* `contaminante` for pixel level modeling
* `allesfitter` for light curve model fitting
* `PyPopStar` for stellar population analysis
* `sedfitter` for SED fitting
* `maelstrom` for pulsating binary analysis
* `stella` & `fleck` for flare detection and modeling
* `eleanor` & `f3` for TESS FFI
* `envelope` for planet's envelope fraction estimation using MESA simulations based on [Milholland+19,20](https://github.com/smillholland/Sub-Saturns.git)
* [`evolstate`](https://github.com/danxhuber/evolstate) for assigning stellar evolutionary state
* [`barycorrpy`](https://github.com/shbhuk/barycorrpy) for conversion to BJD_TDB

## Examples
Check [examples in nbviewer](https://nbviewer.jupyter.org/github/jpdeleon/chronos/tree/master/notebooks/).

## See also
[tql](https://github.com/jpdeleon/tql) for TESS Quick Look plotting

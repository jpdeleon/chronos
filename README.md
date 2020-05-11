# chronos
[![Build Status](https://travis-ci.com/jpdeleon/chronos.svg?branch=master)](https://travis-ci.com/jpdeleon/chronos)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


## Installation
```bash
$ git clone git@github.com:jpdeleon/chronos.git
$ cd chronos && python setup.py install
$ python setup.py develop (optional)
```


## test
```
$ pytest tests/
```


## Modules
* `target.py`: star bookkeeping, e.g. position, catalog cross-matching, archival data look-up
* `star.py`: physics-related calculations, e.g. extinction, spectral typing, isochrones, gyrochronology (inherits `target`)
* `planet.py`: planet parameters calculations (inherits `star`)
* `cluster.py`: cluster catalog, cluster analysis + plotting
* `tpf.py`: targetpixel file manipulation
* `lightcurve.py`: lightcurve analysis either using SPOC or custom pipeline for short and long cadence (inherits `tpf`)
* `k2.py`: tpf and lightcurve for K2; likely to be ingested/refactored to tpf.py & lightcurve.py
* `plot.py`: custom plotting functionalities
* `utils.py`: useful utilities


## Dependencies
* `astropy` & `astroquery` for star bookkeeping
* `lightkurve` & `wotan` for lightcurve analysis
* `emcee` & `corner` for MCMC analysis
* `isochrones` for isochrones analysis
* `dustmass` for extinction calculation
* `stardate` for gyrochronology
* `mrexo` for mass-radius relation
[//]: # * `maelstrom` for pulsating binary analysis
[//]: # * `triceratops` for FPP calculation based on lightcurve shape


## Examples
Check [examples in nbviewer](https://nbviewer.jupyter.org/github/jpdeleon/chronos/tree/master/notebooks/).

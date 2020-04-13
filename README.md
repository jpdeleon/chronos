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


## Modules
* `target.py`: star bookkeeping, e.g. position, catalog cross-matching, archival data look-up
* `star.py`: physics-related calculations, e.g. extinction, spectral typing, isochrones, gyrochronology
* `cluster.py`: cluster catalog, cluster analysis + plotting
* `tpf.py`: targetpixel file manipulation
* `lightcurve.py`: lightcurve analysis either using SPOC or custom pipeline for short and long cadence
* `k2.py`: tpf and lightcurve for K2; likely to be ingested/refactored to tpf.py & lightcurve.py
* `plot.py`: custom plotting functionalities
* `utils.py`: useful utilities


## Examples
Check [examples in nbviewer](https://nbviewer.jupyter.org/github/jpdeleon/chronos/tree/master/notebooks/).

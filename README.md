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
* `lightcurve.py`: light curve analysis either using SPOC, QLP, or custom pipeline for short and long cadence (inherits `tpf`)
* `k2.py`: tpf and light curve for K2; likely to be ingested/refactored to tpf.py & lightcurve.py
* `cluster.py`: cluster catalog, cluster analysis + plotting
* `qlp`, `cdips.py` & `pathos.py`: api for [QLP](http://archive.stsci.edu/hlsp/qlp), [CDIPS](http://archive.stsci.edu/hlsp/qlp), and [PATHOS](http://archive.stsci.edu/hlsp/qlp) pipelines
* `plot.py`: custom plotting functionalities
* `utils.py`: useful utilities

## Dependencies
* `astropy` & `astroquery` for star and catalog bookkeeping
* [`lightkurve`](https://github.com/KeplerGO/lightkurve), [`transitleastsquares`](https://github.com/hippke/tls), & [`wotan`](https://github.com/hippke/wotan) for light curve analysis
* [`emcee`](https://github.com/dfm/emcee) & [`corner`](https://github.com/dfm/corner.py) for MCMC analysis
* [`isochrones`](https://github.com/timothydmorton/isochrones) for isochrones analysis
* [`dustmaps`](https://github.com/gregreen/dustmaps) for extinction calculation
* [`stardate`](https://github.com/RuthAngus/stardate) for gyrochronology
* [`mrexo`](https://github.com/shbhuk/mrexo) for mass-radius relation
* [`triceratops`](https://github.com/stevengiacalone/triceratops) for FPP calculation based on lightcurve shape and contrast curve constraints

## For next update
* [`platon`](https://github.com/ideasrule/platon) for calculating transmission spectrum
* [`theJoker`](https://github.com/adrn/thejoker) for two-body MCMC sampling
* [`spock`](https://github.com/dtamayo/spock), [`dynamite`](https://github.com/JeremyDietrich/dynamite), & [`resonance widths`](https://github.com/katvolk/analytical-resonance-widths) for orbit stability analysis
* [`exoCMD`](https://github.com/gdransfield/ExoCMD) for exoplanet color-magnitude diagrams
* [`contaminante`](https://github.com/christinahedges/contaminante) for pixel level modeling
* [`allesfitter`](https://github.com/MNGuenther/allesfitter) for light curve model fitting
* [`SPISEA`](https://github.com/astropy/SPISEA) for stellar population analysis
* `sedfitter` for SED fitting
* `maelstrom` for pulsating binary analysis
* `stella` & `fleck` for flare detection and modeling
* `eleanor` & `f3` for TESS FFI; See also DIAmante, [Montalto+2020](https://ui.adsabs.harvard.edu/abs/2020arXiv200809832M/abstract)
* `envelope` for planet's envelope fraction estimation using MESA simulations based on [Milholland+19,20](https://github.com/smillholland/Sub-Saturns.git)
* [`evolstate`](https://github.com/danxhuber/evolstate) for assigning stellar evolutionary state
* [`barycorrpy`](https://github.com/shbhuk/barycorrpy) for conversion to BJD_TDB

## Examples
Check [examples in nbviewer](https://nbviewer.jupyter.org/github/jpdeleon/chronos/tree/master/notebooks/).

## TESS exoplanet candidates catalog
* TOI ([csv](https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv))
* CTOI ([csv](https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv))
* CDIPS ([Bouma+2019](https://ui.adsabs.harvard.edu/abs/2019ApJS..245...13B/abstract))
* PATHOS ([Nardiello+2020]("https://ui.adsabs.harvard.edu/abs/2020arXiv200512281N/abstract"))
* DIAmante ([Montalto+2020](https://ui.adsabs.harvard.edu/abs/2020arXiv200809832M/abstract))

## See also
[tql](https://github.com/jpdeleon/tql) for TESS Quick Look plotting

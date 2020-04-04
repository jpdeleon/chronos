# chronos
young stars in associations, moving groups, and star forming regions

## Installation
```bash
$ git clone git@github.com:jpdeleon/chronos.git
$ cd chronos && python setup.py install
$ python setup.py develop (optional)
```


## Modules
* target.py: star bookkeeping, e.g. position, catalog cross-matching, archival data look-up
* star.py: physics-related calculations, e.g. extinction, spectral typing
* cluster.py: cluster catalog, cluster analysis + plotting
* tpf.py: targetpixel file manipulation
* lightcurve.py: lightcurve analysis either using SPOC or custom pipeline for short and long cadence
* age.py: stellar age related estimatation using isochrones, gyrochronology
* k2.py: tpf and lightcurve for K2; likely to be ingested/refactored to tpf.py & lightcurve.py
* plot.py: custom plotting functionalities
* utils.py: useful utilities


## Examples
Check [examples](https://github.com/jpdeleon/chronos/tree/master/notebooks).

# chronos
young stars in associations, moving groups, and star forming regions

## Examples
See more in [examples](https://github.com/jpdeleon/chronos/tree/master/notebooks).

## Scripts
### Cluster membership plots
To find the nearest cluster to TOI 580 (Gaia DR2 5519619186857962112) with cluster membership plots shown:
```bash
./find_cluster_near_target 5519619186857962112 -p
```

![img](./data/Vel_OB2_hrd.png)
![img](./data/Vel_OB2_kinematics.png)
![img](./data/Vel_OB2_xyz_uvw.png)

Try the following Gaia DR2 ids of known planet hosts in a cluster:
* K2-25: 3311804515502788352 in Hyades
* K2-33: 6245758900889486720
* K2-95: 659744295638254336 in NGC_2632 (Praesepe)
* K2-100: 664337230586013312 in NGC_2632 (Praesepe)
* K2-136: 145916050683920128 in Hyades
* K2-264: 661167785238757376 in NGC_2632 (Praesepe)
* K2-284: 3413793491812093824
* CVSO 30: 3222255959210123904 in ASCC_16
* HD 222259A: 6387058411482257536

### CDIPS lightcurves
```bash
./make_cdips_ql 5519619186857962112 -p
```

`chronos` provides an API to download CDIPS lightcurve from [mast](http://archive.stsci.edu/hlsp/cdips).

``` python
from chronos import Target, CDIPS
from chronos.utils import get_toi

toi = 681
t = Target(toiid=toi, verbose=False)
#get gaia id
t.query_gaia_dr2_catalog(return_nearest_xmatch=True)
#initialize cdips
cdips = CDIPS(gaiaDR2id=t.gaiaid, sector=7, aper_idx=3, verbose=False)
#get lc and turn into lk.TessLightCurve
lc = cdips.lc
#add label
lc.label = f'TOI {toi}.01'
#get ephemeris from TOI release (https://tev.mit.edu/data/)
d = get_toi(toi=f'{toi}.01', verbose=False)
per, t0 = d[['Period (days)', 'Epoch (BJD)']].values[0]
#plot
ax = lc.remove_outliers().flatten(window_length=51).fold(period=per, t0=t0).scatter()
ax.set_xlim([-0.05,0.05])
```

![img](./data/cdips_lc.png)

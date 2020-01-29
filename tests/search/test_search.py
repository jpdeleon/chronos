#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import numpy as np

# Import from chronos
from chronos.search import Target, ClusterCatalog

cc = ClusterCatalog()
# Bouma dataset
df = cc.query_catalog(name="Bouma19", return_members=False)
print(df.shape)
assert len(df) > 0
# name
target_coord = Target(name="Pi Mensae").target_coord
assert np.any([target_coord.ra, target_coord.dec])
# k2
target_coord = Target(name="K2-33").target_coord
assert np.any([target_coord.ra, target_coord.dec])
# toi
target_coord1 = Target(toiid=837).target_coord
assert np.all([target_coord1.ra, target_coord1.dec])
# tic
target_coord2 = Target(ticid=460205581).target_coord
assert np.all([target_coord2.ra, target_coord2.dec])
assert target_coord1.ra == target_coord2.ra
assert target_coord1.dec == target_coord2.dec
# also
t = Target(ticid=460205581)
assert len(t.ticid) > 0
assert np.all([t.target_coord.ra, t.target_coord.dec])

# ra,dec
target_coord = Target(ra_deg=157.03729167, dec_deg=-64.50521111).target_coord
assert np.all([target_coord.ra, target_coord.dec])

t = Target(gaiaDR2id=5251470948229949568)
assert len(t.gaiaid) > 0
assert np.all([t.target_coord.ra, t.target_coord.dec])
# epic
t1 = Target(epicid=201270176)
assert len(t1.epicid) > 0
assert np.all([t1.target_coord.ra, t1.target_coord.dec])
t2 = Target(name="EPIC 201270176")
assert np.any([t2.target_coord.ra, t2.target_coord.dec])
assert len(t2.epicid) > 0
assert t1.target_coord.ra == t2.target_coord.ra
assert t1.target_coord.dec == t2.target_coord.dec

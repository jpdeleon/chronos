# -*- coding: utf-8 -*-

# Import modules
import pytest

# Import from package
from chronos.cluster import ClusterCatalog


def test_cluster():
    # catalog
    cc = ClusterCatalog(name="Bouma2019")
    # Bouma dataset
    df = cc.query_catalog(return_members=False)
    assert len(df) > 0
    df_mem = cc.query_catalog(return_members=True)
    assert len(df_mem) > 0
    # cluster

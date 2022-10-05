# -*- coding: utf-8 -*-
"""
test methods of cluster module
"""
import pytest
from chronos.cluster import ClusterCatalog, Cluster

CATALOG = "CantatGaudin2020"
CLUSTER = "IC_2602"


def test_cluster_catalog():
    # catalog
    cc = ClusterCatalog(catalog_name=CATALOG, verbose=False)
    df = cc.query_catalog(return_members=False)
    assert len(df) > 0
    df_mem = cc.query_catalog(return_members=True)
    assert len(df_mem) > 0


@pytest.mark.skip
# FIXME: test takes too long to download
def test_cluster():
    c = Cluster(CLUSTER, catalog_name=CATALOG, verbose=False)
    df_gaia_mem = c.query_cluster_members_gaia_params()
    assert len(df_gaia_mem) > 0

"""
This installation requires git which pulls large files first before pip installation
https://stackoverflow.com/a/58932741/1910174
"""
from setuptools import setup, find_packages
from chronos import __version__, name

import os
import subprocess
import sys

setup(
    name=name,
    version=__version__,
    description="exploring young star catalogs",
    url="http://github.com/jpdeleon/chronos",
    author="Jerome de Leon",
    author_email="jpdeleon.bsap@gmail.com",
    license="MIT",
    packages=find_packages(),
    scripts=[
        "scripts/make_tql",
        "scripts/rank_tls",
        "scripts/make_tql_per_cluster",
        "scripts/query_cluster_members_gaia_params",
        "scripts/find_cluster_near_target",
        "scripts/make_cdips_ql",
    ],
    install_requires=[
        "astroquery==0.4",
        "lightkurve==1.9.0",
        "astropy==4.0",
        "pandas==1.0.1",
        "tqdm",
        "astroplan==0.6",
        "transitleastsquares",
        # wotan
        "scikit-image==0.16.2",  # just for measure.find_contours
        # pprint==3.8.2
        "deepdish==0.3.6",
    ],
)

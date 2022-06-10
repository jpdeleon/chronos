"""
This installation requires git which pulls large files first before pip installation
https://stackoverflow.com/a/58932741/1910174
"""
from setuptools import setup, find_packages
import sys

if not (sys.version_info.major == 3) & (sys.version_info.minor == 6):
    sys.exit("Sorry, this package only works using Python 3.6")

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = fh.read().splitlines()

setup(
    name="chronos",
    version="1.0.0",
    # python_requires='>3.6.1,<3.6.13',
    description="toolkit for discovery and characterization of exoplanets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/jpdeleon/chronos",
    author="Jerome de Leon",
    author_email="jpdeleon.bsap@gmail.com",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    # package_data={'chronos': ['data/*.csv']},
    # scripts=[
    #    "calc_fpp",
    #     "scripts/check_target_in_cluster",
    #     "scripts/make_tql_per_cluster",
    #     "scripts/query_cluster_members_gaia_params",
    #     "scripts/find_cluster_near_target",
    #     "scripts/make_cdips_ql",
    # ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    extras_requires={
        "triceratops": [
            "git+https://github.com/stevengiacalone/triceratops.git#egg=triceratops"
        ],
        "contaminante": [
            "git+https://github.com/christinahedges/contaminante#egg=contaminante"
        ],
        "spisea": ["https://github.com/astropy/SPISEA#egg=PyPopStar"],
        "isochrones": [
            "git+https://github.com/timothydmorton/isochrones.git#egg=isochrones"
        ],
        "stardate": [
            "git+https://github.com/RuthAngus/stardate.git#egg=stardate"
        ],
        "dustmaps": [
            "git+https://github.com/gregreen/dustmaps.git#egg=dustmaps"
        ],
        "corner": ["git+https://github.com/dfm/corner.py.git#egg=corner"],
        "maelstrom": [
            "git+https://github.com/danhey/maelstrom.git#egg=maelstrom"
        ],
        "fleck": ["git+https://github.com/bmorris3/fleck.git#egg=fleck"],
    },
)

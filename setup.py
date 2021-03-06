"""
This installation requires git which pulls large files first before pip installation
https://stackoverflow.com/a/58932741/1910174
"""
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = fh.read().splitlines()

setup(
    name="chronos",
    version="0.1.0",
    description="discovery and characterization of young stars",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/jpdeleon/chronos",
    author="Jerome de Leon",
    author_email="jpdeleon.bsap@gmail.com",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
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
        "popstar": [
            "git+https://github.com/astropy/PyPopStar.git#egg=PyPopStar"
        ],
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
